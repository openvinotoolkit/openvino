# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from packaging import version

from openvino import Core, convert_model
from openvino.frontend import OpConversionFailure
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder


pytestmark = [
    pytest.mark.precommit,
    pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.11"),
                       reason="Requires flat_apply in torch.export"),
]


def _flat_apply_graph(output_kind, dynamic):
    from torch._higher_order_ops.flat_apply import flat_apply, func_to_graphable
    from torch.utils._pytree import tree_flatten

    class Inputs(torch.nn.Module):
        def forward(self, x, y, scale):
            return x, y, scale

    def function(pair, *, scale):
        x, bias, y = pair
        result = x + y * scale + bias
        if output_kind == "tensor":
            return result
        if output_kind == "singleton":
            return (result,)
        return result, [x.sin(), (y.cos(),)]

    examples = (torch.randn(3, 4), torch.randn(3, 4), torch.tensor(2.))
    dynamic_shapes = None
    if dynamic:
        batch = torch.export.Dim("batch", min=2)
        dynamic_shapes = ({0: batch}, {0: batch}, {})
    exported = torch.export.export(Inputs(), examples, dynamic_shapes=dynamic_shapes)
    operands = [node for node in exported.graph.nodes if node.op == "placeholder"]
    graph = torch.fx.Graph()
    inputs = [graph.placeholder(node.name) for node in operands]
    for node, original in zip(inputs, operands):
        node.meta = original.meta.copy()
    flat_args, input_spec = tree_flatten((((inputs[0], 3, inputs[1]),), {"scale": inputs[2]}))
    _, function_spec = func_to_graphable(function)
    result = graph.call_function(flat_apply, (graph.get_attr("function_spec"),
                                             graph.get_attr("input_spec"), *flat_args))
    if output_kind == "tensor":
        outputs = (result,)
    elif output_kind == "singleton":
        outputs = (graph.call_function(operator.getitem, (result, 0)),)
    else:
        first = graph.call_function(operator.getitem, (result, 0))
        nested = graph.call_function(operator.getitem, (result, 1))
        second = graph.call_function(operator.getitem, (nested, 0))
        nested_tuple = graph.call_function(operator.getitem, (nested, 1))
        third = graph.call_function(operator.getitem, (nested_tuple, 0))
        outputs = (first, second, third)
    graph.output(outputs)
    return torch.fx.GraphModule({"function_spec": function_spec, "input_spec": input_spec}, graph), examples


@pytest.mark.parametrize("output_kind", ["tensor", "singleton", "nested"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_flat_apply_tensor_inputs_and_nested_outputs(output_kind, dynamic):
    model, examples = _flat_apply_graph(output_kind, dynamic)
    original_graph = str(model.graph)
    with patch.object(torch.export.ExportedProgram, "run_decompositions",
                      side_effect=AssertionError("Conversion must not run decompositions")), \
            patch.object(torch.export, "export", side_effect=AssertionError("Conversion must not retrace the model")):
        converted = convert_model(TorchFXPythonDecoder(model, dynamic_shapes=dynamic))
    compiled = Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    assert str(model.graph) == original_graph
    scenarios = [examples, (torch.randn(5 if dynamic else 3, 4),
                            torch.randn(5 if dynamic else 3, 4), torch.tensor(-1.5))]
    for inputs in scenarios:
        actual = compiled([value.numpy() for value in inputs])
        expected = model(*inputs)
        assert len(actual) == len(expected)
        for index, value in enumerate(expected):
            np.testing.assert_allclose(actual[index], value.numpy(), atol=1e-6, rtol=1e-5)


def test_flat_apply_custom_autograd_function():
    from torch._dynamo._trace_wrapped_higher_order_op import mod_index
    from torch._higher_order_ops.flat_apply import flat_apply, func_to_graphable
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.export.custom_ops import _call_custom_autograd_function_in_pre_dispatch
    from torch.utils._pytree import tree_flatten

    inputs = (torch.arange(12, dtype=torch.float32).reshape(3, 4),
              torch.tensor([0, 2]), torch.tensor([1, 3]))
    graph = torch.fx.Graph()
    operands = [graph.placeholder(name) for name in ("x", "i", "j")]
    with FakeTensorMode() as mode:
        for operand, value in zip(operands, inputs):
            operand.meta["val"] = mode.from_tensor(value)
    flat_args, input_spec = tree_flatten((("torch._dynamo._trace_wrapped_higher_order_op.ModIndex",
                                         operands[0], operands[1:]), {}))
    _, function_spec = func_to_graphable(_call_custom_autograd_function_in_pre_dispatch)
    result = graph.call_function(flat_apply, (graph.get_attr("function_spec"),
                                             graph.get_attr("input_spec"), *flat_args))
    graph.output((result,))
    model = torch.fx.GraphModule({"function_spec": function_spec, "input_spec": input_spec}, graph)
    original_graph = str(model.graph)
    with patch.object(torch.export.ExportedProgram, "run_decompositions",
                      side_effect=AssertionError("Conversion must not run decompositions")), \
            patch.object(torch.export, "export", side_effect=AssertionError("Conversion must not retrace the model")):
        converted = convert_model(TorchFXPythonDecoder(model))
    compiled = Core().compile_model(converted, "CPU")
    assert str(model.graph) == original_graph
    for values in (inputs, (inputs[0], torch.tensor([1, 0]), torch.tensor([2, 0]))):
        expected = mod_index(values[0], list(values[1:]))
        np.testing.assert_array_equal(compiled([value.numpy() for value in values])[0], expected.numpy())


def test_flat_apply_symbolic_operand():
    from torch._higher_order_ops.flat_apply import flat_apply, func_to_graphable
    from torch.utils._pytree import tree_flatten

    class Model(torch.nn.Module):
        def forward(self, x):
            return x, x.shape[0]

    def function(x, size):
        return x + size

    exported = torch.export.export(Model(), (torch.ones(3, 4),),
                                   dynamic_shapes=({0: torch.export.Dim("batch", min=2)},))
    source_nodes = list(exported.graph.nodes)
    source_output = next(node for node in source_nodes if node.op == "output")
    graph = torch.fx.Graph()
    mapping = {}
    for node in source_nodes:
        if node.op != "output":
            mapping[node] = graph.node_copy(node, lambda value: mapping[value])
    x, size = [mapping[node] for node in source_output.args[0]]
    _, function_spec = func_to_graphable(function)
    _, input_spec = tree_flatten(((None, None), {}))
    result = graph.call_function(flat_apply, (graph.get_attr("function_spec"),
                                             graph.get_attr("input_spec"), x, size))
    graph.output((result,))
    model = torch.fx.GraphModule({"function_spec": function_spec, "input_spec": input_spec}, graph)
    compiled = Core().compile_model(convert_model(TorchFXPythonDecoder(model, dynamic_shapes=True)), "CPU")
    for batch in [3, 5]:
        data = torch.ones(batch, 4)
        np.testing.assert_array_equal(compiled([data.numpy()])[0], (data + batch).numpy())


def test_flat_apply_logical_batched_tensor_metadata():
    from torch._subclasses.fake_tensor import FakeTensorMode

    functorch = torch._C._functorch
    with FakeTensorMode():
        physical = torch.empty(2, 3, 4)
    level = functorch._vmap_increment_nesting(2, "error")
    try:
        batched = functorch._add_batch_dim(physical, 0, level)
    finally:
        functorch._vmap_decrement_nesting()
    example = TorchFXPythonDecoder._flat_apply_example(batched)
    assert not functorch.is_batchedtensor(example)
    assert example.shape == (3, 4)
    assert example.dtype == physical.dtype


@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "nested", "mutated_body"])
def test_flat_apply_output_alias_mutation(output_kind):
    from torch._higher_order_ops.flat_apply import flat_apply, func_to_graphable
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._pytree import tree_flatten

    def function(x):
        view = x[:, 1:]
        if output_kind == "mutated_body":
            view.add_(2)
            return x * 3
        if output_kind == "tensor":
            return view
        if output_kind == "tuple":
            return (view,)
        return [x.sin(), [view]]

    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    graph = torch.fx.Graph()
    operand = graph.placeholder("x")
    value = graph.call_function(torch.ops.aten.clone.default, (operand,))
    with FakeTensorMode() as mode:
        operand.meta["val"] = mode.from_tensor(data)
        value.meta["val"] = operand.meta["val"].clone()
    _, function_spec = func_to_graphable(function)
    _, input_spec = tree_flatten(((None,), {}))
    result = graph.call_function(flat_apply, (graph.get_attr("function_spec"),
                                             graph.get_attr("input_spec"), value))
    view = result
    if output_kind == "tuple":
        view = graph.call_function(operator.getitem, (result, 0))
    elif output_kind == "nested":
        nested = graph.call_function(operator.getitem, (result, 1))
        view = graph.call_function(operator.getitem, (nested, 0))
    graph.call_function(torch.ops.aten.add_.Tensor, (view, 2))
    graph.output((value, view))
    model = torch.fx.GraphModule({"function_spec": function_spec, "input_spec": input_spec}, graph)
    decoder = TorchFXPythonDecoder(model)
    if output_kind == "nested":
        with pytest.raises(OpConversionFailure,
                           match="flat_apply does not support aliased operands in nested output containers"):
            convert_model(decoder)
        return
    compiled = Core().compile_model(convert_model(decoder), "CPU")
    actual = compiled([data.numpy()])
    expected = model(data)
    for index, value in enumerate(expected):
        np.testing.assert_array_equal(actual[index], value.numpy())


@pytest.mark.parametrize("specialization", ["data", "shape"])
def test_flat_apply_rejects_runtime_specialization(specialization):
    from torch._higher_order_ops.flat_apply import func_to_graphable

    model, _ = _flat_apply_graph("tensor", specialization == "shape")

    def data_dependent(pair, *, scale):
        x, _, y = pair
        condition = scale.item() > 0 if specialization == "data" else x.shape[0] > 3
        return x if condition else y

    _, model.function_spec = func_to_graphable(data_dependent)
    decoder = TorchFXPythonDecoder(model)
    with pytest.raises(RuntimeError, match="Cannot trace flat_apply callable without specializing runtime data"):
        decoder.visit_subgraph(lambda node: None)


def test_flat_apply_legacy_guard_context(monkeypatch):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    size = shape_env.create_unbacked_symint()
    monkeypatch.setattr(shape_env, "error_on_new_guards", None, raising=False)
    original_frozen = shape_env.frozen
    original_asserts = shape_env.num_deferred_runtime_asserts
    original_guards = len(shape_env.guards)
    with pytest.raises(RuntimeError, match="new runtime shape guard"):
        with TorchFXPythonDecoder._flat_apply_no_new_guards(shape_env):
            shape_env.guard_or_defer_runtime_assert(size.node.expr > 0, "size must be positive")
    assert shape_env.frozen == original_frozen
    assert "_check_frozen" not in shape_env.__dict__
    assert shape_env.num_deferred_runtime_asserts == original_asserts
    assert len(shape_env.guards) == original_guards
