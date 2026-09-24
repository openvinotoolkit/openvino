# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from packaging import version

from openvino import Core, Type, convert_model
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder


pytestmark = [
    pytest.mark.precommit,
    pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.9"),
                       reason="Requires grad-mode wrappers in torch.export"),
]


def convert_without_decompositions(exported):
    original_graphs = {
        name: str(module.graph)
        for name, module in exported.graph_module.named_modules()
        if isinstance(module, torch.fx.GraphModule)
    }
    with patch.object(torch.export.ExportedProgram, "run_decompositions",
                      side_effect=AssertionError("Conversion must not run decompositions")):
        converted = convert_model(exported)
    for name, graph in original_graphs.items():
        assert str(exported.graph_module.get_submodule(name).graph) == graph
    return Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


@pytest.mark.parametrize("grad_enabled", [False, True])
@pytest.mark.parametrize("multiple_outputs", [False, True])
def test_export_grad_mode(grad_enabled, multiple_outputs):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.arange(1, 5, dtype=torch.float32))

        @torch.set_grad_enabled(grad_enabled)
        def rotary(self, x, positions):
            frequencies = self.inv_freq[None, :, None] @ positions[:, None, :].float()
            frequencies = frequencies.transpose(1, 2)
            return frequencies.cos() + x, frequencies.sin() + x

        def forward(self, data):
            cosine, sine = self.rotary(data["x"], data["positions"])
            return (cosine, sine) if multiple_outputs else sine

    model = Model().eval()
    data = {"x": torch.ones(2, 8, 4), "positions": torch.arange(8).expand(2, -1)}
    batch, sequence = torch.export.Dim("batch"), torch.export.Dim("sequence")
    dynamic = {"data": {key: {0: batch, 1: sequence} for key in data}}
    with torch.set_grad_enabled(not grad_enabled):
        exported = torch.export.export(model, (data,), dynamic_shapes=dynamic)
    assert any(str(node.target) == "wrap_with_set_grad_enabled" for node in exported.graph.nodes)
    compiled = convert_without_decompositions(exported)
    for batch_size, sequence_length in [(2, 8), (3, 5)]:
        inputs = {
            "x": torch.ones(batch_size, sequence_length, 4),
            "positions": torch.arange(sequence_length).expand(batch_size, -1),
        }
        with torch.no_grad():
            expected = model(inputs)
        if not multiple_outputs:
            expected = (expected,)
        actual = compiled([value.numpy() for value in inputs.values()])
        assert len(actual) == len(expected)
        for index, value in enumerate(expected):
            np.testing.assert_allclose(actual[index], value.numpy(), atol=1e-5, rtol=1e-5)


def test_export_grad_mode_mutations():
    class Model(torch.nn.Module):
        @torch.no_grad()
        def update(self, x):
            view = x[:, 1:]
            view.add_(2)
            with torch.enable_grad():
                before = view.sin()
            return view, before

        def forward(self, x):
            x = x.clone()
            view, before = self.update(x)
            view.add_(1)
            return x, view, before

    model = Model().eval()
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    with torch.enable_grad():
        exported = torch.export.export(model, (data,))
    wrappers = [node for node in exported.graph.nodes
                if str(node.target) == "wrap_with_set_grad_enabled"]
    assert wrappers
    compiled = convert_without_decompositions(exported)
    expected = model(data)
    actual = compiled([data.numpy()])
    assert len(actual) == len(expected)
    for index, value in enumerate(expected):
        np.testing.assert_allclose(actual[index], value.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mutate_base", [False, True])
def test_export_autocast_returns_view(mutate_base):
    class Model(torch.nn.Module):
        def forward(self, x):
            x = x.clone()
            with torch.autocast("cpu", enabled=False):
                view = x.transpose(0, 1)[1:]
                view.add_(2)
                before = view * 2
            if mutate_base:
                x.mul_(3)
            else:
                view.add_(1)
            return x, view, before

    model = Model().eval()
    data = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    exported = torch.export.export(model, (data,))
    assert any(str(node.target) == "wrap_with_autocast" for node in exported.graph.nodes)
    compiled = convert_without_decompositions(exported)
    expected = model(data)
    actual = compiled([data.numpy()])
    assert len(actual) == len(expected)
    for index, value in enumerate(expected):
        np.testing.assert_allclose(actual[index], value.numpy(), atol=1e-5, rtol=1e-5)


def test_export_enabled_autocast_output_types():
    class Model(torch.nn.Module):
        def forward(self, x, w):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                y = x @ w
                z = y + x
            return y, z

    model = Model().eval()
    data = (torch.randn(4, 4), torch.randn(4, 4))
    exported = torch.export.export(model, data)
    assert any(str(node.target) == "wrap_with_autocast" for node in exported.graph.nodes)
    ov_model = convert_model(exported)
    expected = model(*data)
    assert [output.get_element_type() for output in ov_model.outputs] == [Type.bf16, Type.f32]
    actual = Core().compile_model(ov_model, "CPU")([value.numpy() for value in data])
    # bf16 results are returned as raw 16-bit data.
    y = (actual[0].view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    np.testing.assert_allclose(y, expected[0].float().numpy(), atol=0.05, rtol=0.02)
    np.testing.assert_allclose(actual[1], expected[1].numpy(), atol=0.05, rtol=0.02)


def test_export_grad_mode_unsupported_view_mutation():
    class Model(torch.nn.Module):
        def forward(self, x):
            x = x.clone()
            with torch.no_grad():
                view = x.diagonal()
            view.add_(1)
            return x

    data = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    exported = torch.export.export(Model().eval(), (data,))
    # The mutation cannot be propagated to the operand, so conversion must fail instead of dropping it.
    with pytest.raises(Exception, match="wrap_with_set_grad_enabled_reverseprop"):
        convert_model(exported)


def test_fx_nested_grad_mode():
    from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

    body_graph = torch.fx.Graph()
    data = body_graph.placeholder("x")
    offset = body_graph.get_attr("offset")
    result = body_graph.call_function(torch.ops.aten.add.Tensor, (data, offset))
    body_graph.output((result,))
    body = torch.fx.GraphModule({"offset": torch.tensor([2.0, 3.0])}, body_graph)

    def wrap(module, enabled):
        graph = torch.fx.Graph()
        data = graph.placeholder("x")
        body_node = graph.get_attr("body")
        result = graph.call_function(wrap_with_set_grad_enabled, (enabled, body_node, data))
        graph.output(result)
        return torch.fx.GraphModule({"body": module}, graph)

    model = wrap(wrap(body, False), True)
    original = str(model.graph)
    data = torch.ones(2)
    with patch.object(torch.export.ExportedProgram, "run_decompositions",
                      side_effect=AssertionError("Conversion must not run decompositions")):
        decoder = TorchFXPythonDecoder(model, input_shapes=[data.shape], input_types=[data.dtype])
        converted = convert_model(decoder)
    compiled = Core().compile_model(converted, "CPU")
    actual = compiled([data.numpy()])
    np.testing.assert_array_equal(actual[0], model(data)[0].numpy())
    assert str(model.graph) == original


def test_fx_grad_mode_returns_operand():
    from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

    body_graph = torch.fx.Graph()
    body_graph.output((body_graph.placeholder("x"),))
    body = torch.fx.GraphModule({}, body_graph)

    # torch.export does not return operands from the body, but FX graphs may.
    graph = torch.fx.Graph()
    data = graph.placeholder("x")
    data = graph.call_function(torch.ops.aten.clone.default, (data,))
    wrapped = graph.call_function(wrap_with_set_grad_enabled, (False, graph.get_attr("body"), data))
    returned = graph.call_function(operator.getitem, (wrapped, 0))
    graph.call_function(torch.ops.aten.add_.Tensor, (returned, 1))
    graph.output((data,))
    model = torch.fx.GraphModule({"body": body}, graph)

    data = torch.zeros(2)
    decoder = TorchFXPythonDecoder(model, input_shapes=[data.shape], input_types=[data.dtype])
    compiled = Core().compile_model(convert_model(decoder), "CPU")
    np.testing.assert_array_equal(compiled([data.numpy()])[0], model(data)[0].numpy())
