# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.14"),
                       reason="Requires predispatch vmap operators in torch.export"),
]


def _convert_export(model, inputs, dynamic_shapes=None):
    exported = torch.export.export(model, tuple(value.clone() for value in inputs),
                                   dynamic_shapes=dynamic_shapes)
    graphs = {name: str(module.graph) for name, module in exported.graph_module.named_modules()
              if isinstance(module, torch.fx.GraphModule)}
    with patch.object(torch.export.ExportedProgram, "run_decompositions",
                      side_effect=AssertionError("Conversion must not run decompositions")):
        converted = convert_model(exported)
    for name, graph in graphs.items():
        assert str(exported.graph_module.get_submodule(name).graph) == graph
    return Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


def _compare(compiled, model, inputs):
    expected = model(*inputs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = compiled([value.numpy() for value in inputs])
    for i, value in enumerate(expected):
        np.testing.assert_allclose(actual[i], value.numpy(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("in_dim", [1, -2])
@pytest.mark.parametrize("out_dim", [0])
def test_export_vmap_dimensions(in_dim, out_dim):
    class Model(torch.nn.Module):
        def forward(self, data):
            return torch.vmap(lambda row: row.sin().sum(dim=-1),
                              in_dims=in_dim, out_dims=out_dim)(data)

    model = Model()
    inputs = (torch.randn(2, 4, 3),)
    batch = torch.export.Dim("batch", min=2, max=16)
    compiled = _convert_export(model, inputs, ({1: batch},))
    for size in (4, 6):
        _compare(compiled, model, (torch.randn(2, size, 3),))


def test_export_vmap_multiple_inputs_and_outputs():
    class Model(torch.nn.Module):
        def forward(self, data, bias):
            return torch.vmap(lambda row, offset: ((row + offset).sum(-1), row.cos()),
                              in_dims=(0, 0), out_dims=(0, 0))(data, bias)

    model = Model()
    batch = torch.export.Dim("batch", min=2, max=16)
    compiled = _convert_export(model, (torch.randn(4, 2, 3), torch.randn(4, 3)),
                               ({0: batch}, {0: batch}))
    for size in (4, 6):
        _compare(compiled, model, (torch.randn(size, 2, 3), torch.randn(size, 3)))


@pytest.mark.parametrize("in_dim", [1, -2])
@pytest.mark.parametrize("out_dim", [1, -1])
def test_fx_vmap_nonleading_output(in_dim, out_dim):
    from torch._functorch import predispatch

    class Model(torch.nn.Module):
        def forward(self, data, bias):
            return torch.vmap(lambda row, offset: ((row + offset).sum(-1), row.cos()),
                              in_dims=(in_dim, 0), out_dims=(out_dim, 1))(data, bias)

    # PyTorch 2.14 export lifts a fake constant when out_dim is nonzero.
    graph = torch.fx.Graph()
    data = graph.placeholder("data")
    bias = graph.placeholder("bias")
    size = graph.call_function(torch.ops.aten.sym_size.int, (data, 1))
    graph.call_function(predispatch._vmap_increment_nesting, (size, "error"))
    row = graph.call_function(predispatch._add_batch_dim, (data, in_dim, 1))
    offset = graph.call_function(predispatch._add_batch_dim, (bias, 0, 1))
    added = graph.call_function(torch.ops.aten.add.Tensor, (row, offset))
    summed = graph.call_function(torch.ops.aten.sum.dim_IntList, (added, [-1]))
    cosine = graph.call_function(torch.ops.aten.cos.default, (row,))
    first = graph.call_function(predispatch._remove_batch_dim, (summed, 1, size, out_dim))
    second = graph.call_function(predispatch._remove_batch_dim, (cosine, 1, size, 1))
    graph.call_function(predispatch._vmap_decrement_nesting, ())
    graph.output((first, second))
    module = torch.fx.GraphModule({}, graph)
    original = str(graph)
    decoder = TorchFXPythonDecoder(module, input_shapes=[[2, -1, 3], [-1, 3]],
                                   input_types=[torch.float32, torch.float32])
    with patch.object(torch.export.ExportedProgram, "run_decompositions",
                      side_effect=AssertionError("Conversion must not run decompositions")):
        compiled = Core().compile_model(convert_model(decoder), "CPU",
                                         {"INFERENCE_PRECISION_HINT": "f32"})
    assert str(module.graph) == original
    for size in (0, 4, 6):
        _compare(compiled, Model(), (torch.randn(2, size, 3), torch.randn(size, 3)))


def test_export_vmap_nested():
    class Model(torch.nn.Module):
        def forward(self, data):
            return torch.vmap(torch.vmap(lambda row: row.sin().sum(-1)))(data)

    model = Model()
    batch = torch.export.Dim("batch", min=2, max=16)
    compiled = _convert_export(model, (torch.randn(2, 3, 4),), ({1: batch},))
    for size in (3, 5):
        _compare(compiled, model, (torch.randn(2, size, 4),))


@pytest.mark.parametrize("out_dim", [0, -1])
def test_export_vmap_invariant_output(out_dim):
    class Model(torch.nn.Module):
        def forward(self, data, captured):
            return torch.vmap(lambda row: captured.sin(), out_dims=out_dim)(data)

    model = Model()
    batch = torch.export.Dim("batch", min=2, max=16)
    compiled = _convert_export(model, (torch.randn(4, 2), torch.randn(3)),
                               ({0: batch}, {}))
    for size in (4, 6):
        _compare(compiled, model, (torch.randn(size, 2), torch.randn(3)))


def test_export_vmap_nested_mask():
    class Model(torch.nn.Module):
        def forward(self, groups):
            indices = torch.arange(groups.shape[1])

            def mask(batch, head, query, key):
                return (key <= query) & (groups[batch, query] == groups[batch, key])

            mapped = torch.vmap(mask, in_dims=(None, None, None, 0))
            mapped = torch.vmap(mapped, in_dims=(None, None, 0, None))
            mapped = torch.vmap(mapped, in_dims=(None, 0, None, None))
            mapped = torch.vmap(mapped, in_dims=(0, None, None, None))
            return mapped(torch.arange(groups.shape[0]), torch.arange(1), indices, indices)

    model = Model()
    sequence = torch.export.Dim("sequence", min=2, max=16)
    compiled = _convert_export(model, (torch.tensor([[0, 0, 1, 1, 1]]),), ({1: sequence},))
    for values in ([[0, 0, 1, 1, 1]], [[0, 1, 1]]):
        _compare(compiled, model, (torch.tensor(values),))


@pytest.mark.parametrize("randomness", ["same", "different"])
def test_export_vmap_rejects_randomness(randomness):
    class Model(torch.nn.Module):
        def forward(self, data):
            return torch.vmap(lambda row: row.sin(), randomness=randomness)(data)

    with pytest.raises(OpConversionFailure, match="vmap supports only randomness='error'"):
        _convert_export(Model(), (torch.randn(2, 4),))


@pytest.mark.parametrize("captured_mutation", [False, True])
def test_export_vmap_rejects_mutation(captured_mutation):
    class Model(torch.nn.Module):
        def forward(self, data):
            captured = torch.zeros_like(data[0])

            def mutate(row):
                if captured_mutation:
                    captured.add_(1)
                    return row + captured
                return row.clone().add_(1)

            return torch.vmap(mutate)(data)

    with pytest.raises(OpConversionFailure, match="Mutation inside vmap is not supported"):
        _convert_export(Model(), (torch.randn(2, 4),))
