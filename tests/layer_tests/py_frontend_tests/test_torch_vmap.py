# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import numpy as np
import pytest
import torch
from packaging import version

from openvino import Core, convert_model
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder


pytestmark = [
    pytest.mark.precommit,
    pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.14"),
                       reason="Requires torch 2.14 or newer"),
]


def _convert_export(model, inputs, dynamic_shapes=None):
    exported = torch.export.export(model, tuple(value.clone() for value in inputs), dynamic_shapes=dynamic_shapes)
    graphs = {name: str(module.graph) for name, module in exported.graph_module.named_modules()
              if isinstance(module, torch.fx.GraphModule)}
    with patch.object(torch.export.ExportedProgram, "run_decompositions",
                      side_effect=AssertionError("Conversion must not run decompositions")):
        converted = convert_model(exported)
    assert not any(node.get_type_name() == "Loop" for node in converted.get_ordered_ops())
    for name, graph in graphs.items():
        assert str(exported.graph_module.get_submodule(name).graph) == graph
    return Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})


def _compare(compiled, model, inputs):
    expected = model(*inputs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = compiled([value.numpy() for value in inputs])
    for i, value in enumerate(expected):
        reference = value.numpy()
        assert actual[i].dtype == reference.dtype
        np.testing.assert_allclose(actual[i], reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("out_dim", [0, 1, -1])
def test_export_vmap_dimensions(out_dim):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.tensor([1., 2., 3.]))

        def forward(self, data, bias, captured):
            # Torch 2.14 export supports only these out_dims for the batched outputs.
            return torch.vmap(lambda row, offset: ((row + offset * self.weight).sum(-1), row.cos(), captured.sin()),
                              in_dims=(1, 0), out_dims=(0, 1, out_dim))(data, bias)

    model = Model()
    batch = torch.export.Dim("batch", min=2, max=16)
    compiled = _convert_export(model, (torch.randn(2, 4, 3), torch.randn(4, 3), torch.randn(5)),
                               ({1: batch}, {0: batch}, {}))
    for size in (0, 4, 6):
        _compare(compiled, model, (torch.randn(2, size, 3), torch.randn(size, 3), torch.randn(5)))


@pytest.mark.parametrize("mod_index", [False, True])
def test_export_vmap_nested_mask(mod_index):
    # Transformers masking indexes inside vmap through TransformGetItemToIndex, which exports flat_apply.
    from contextlib import nullcontext
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    class Model(torch.nn.Module):
        def forward(self, groups):
            indices = torch.arange(groups.shape[1])

            def mask(batch, head, query, key):
                return (key <= query) & (groups[batch, query] == groups[batch, key])

            mapped = torch.vmap(mask, in_dims=(None, None, None, 0))
            mapped = torch.vmap(mapped, in_dims=(None, None, 0, None))
            mapped = torch.vmap(mapped, in_dims=(None, 0, None, None))
            mapped = torch.vmap(mapped, in_dims=(0, None, None, None))
            with TransformGetItemToIndex() if mod_index else nullcontext():
                return mapped(torch.arange(groups.shape[0]), torch.arange(1), indices, indices)

    model = Model()
    sequence = torch.export.Dim("sequence", min=2, max=16)
    compiled = _convert_export(model, (torch.tensor([[0, 0, 1, 1, 1]]),), ({1: sequence},))
    for values in ([[0, 0, 1, 1, 1]], [[0, 1, 1]]):
        _compare(compiled, model, (torch.tensor(values),))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_model_vmap_preserves_export_capture(nested, dynamic):
    import torch.export._trace as export_trace
    import torch.fx.experimental.proxy_tensor as proxy_tensor

    class Model(torch.nn.Module):
        @torch.no_grad()
        def forward(self, data):
            difference = torch.diff(data, dim=-1)
            mapped = torch.vmap(lambda row: row.sin().sum(-1))
            if nested:
                mapped = torch.vmap(mapped)
            return mapped(difference), difference

    model = Model()
    inputs = (torch.randn(2, 4, 3),)
    shapes = ({1: torch.export.Dim("batch", min=2, max=16)},) if dynamic else None
    with patch.object(export_trace, "make_fx", wraps=export_trace.make_fx) as export_tracer, \
            patch.object(proxy_tensor, "make_fx", wraps=proxy_tensor.make_fx) as region_tracer, \
            patch.object(torch.export.ExportedProgram, "run_decompositions",
                         side_effect=AssertionError("Conversion must not run decompositions")):
        decoder = TorchFXPythonDecoder.from_model(model, inputs, dynamic_shapes=shapes)
        converted = convert_model(decoder)
    assert export_tracer.call_count == 1
    assert region_tracer.call_count > 0
    targets = [str(node.target) for module in decoder.pt_module.modules()
               if isinstance(module, torch.fx.GraphModule) for node in module.graph.nodes]
    assert any("aten.diff" in target for target in targets)
    assert any("wrap_with_set_grad_enabled" in target for target in targets)
    # The vmap region is lowered on a copy of the no_grad body, leaving the exported body unchanged.
    assert any("_add_batch_dim" in target for target in targets)
    assert not any(node.get_type_name() == "Loop" for node in converted.get_ordered_ops())
    compiled = Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for size in ((4, 6) if dynamic else (4,)):
        _compare(compiled, model, (torch.randn(2, size, 3),))


def test_model_vmap_module_extension():
    from openvino.frontend.pytorch import ModuleExtension

    class Cosine(torch.nn.Module):
        def forward(self, data):
            return data.cos()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cosine = Cosine()

        def forward(self, data):
            return torch.vmap(lambda row: row.sin())(self.cosine(data))

    data = torch.randn(3, 4)
    model = Model()
    converted = convert_model(model, example_input=(data,), dynamo=True,
                              extension=[ModuleExtension(Cosine, "aten.sin.default")])
    assert not any(node.get_type_name() == "Loop" for node in converted.get_ordered_ops())
    assert not hasattr(model.cosine, "_openvino_module_extension_patch_orig_forward")
    compiled = Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    np.testing.assert_allclose(compiled(data.numpy())[0], data.sin().sin().numpy(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("view_inside", [False, True])
def test_export_vmap_output_alias_mutation(view_inside):
    class Model(torch.nn.Module):
        def forward(self, data):
            if view_inside:
                def mapped(row):
                    value = row.sin()
                    return value.view(2, 4), value.view(-1)

                value, view = torch.vmap(mapped)(data)
            else:
                value = torch.vmap(torch.sin)(data)
                view = value.view(-1)
            view.add_(1)
            return value, view

    model = Model()
    data = torch.randn(3, 2, 4)
    batch = torch.export.Dim("batch", min=2, max=8)
    compiled = _convert_export(model, (data,), ({0: batch},))
    for size in (3, 5):
        _compare(compiled, model, (torch.randn(size, 2, 4),))


def test_export_vmap_cond_unmapped_predicate():
    class Model(torch.nn.Module):
        def forward(self, data, predicate):
            return torch.vmap(lambda row: torch.ops.higher_order.cond(
                predicate, lambda x: (x.sin(),), lambda x: (x.cos(),), (row,)))(data)[0]

    model = Model()
    data = torch.randn(3, 4)
    compiled = _convert_export(model, (data, torch.tensor(True)))
    for predicate in (True, False):
        _compare(compiled, model, (data, torch.tensor(predicate)))


@pytest.mark.parametrize("complex_first", [False, True])
def test_export_vmap_mixed_dtype_aliases(complex_first):
    from openvino.frontend.pytorch import vmap_lowering

    class Model(torch.nn.Module):
        def forward(self, data):
            def mapped(row):
                value = torch.view_as_complex(row.sin()[1:])
                real = torch.view_as_real(value)
                return (value, real) if complex_first else (real, value)

            outputs = torch.vmap(mapped)(data)
            value, real = outputs if complex_first else outputs[::-1]
            return value.real, value.imag, real

    model = Model()
    batch = torch.export.Dim("batch", min=2, max=8)
    exported = torch.export.export(model, (torch.randn(3, 6, 2),), dynamic_shapes=({0: batch},))
    metadata = {}
    trace = vmap_lowering._trace

    def capture(*args):
        traced = trace(*args)
        for node in traced.graph.nodes:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                metadata[node] = (value.dtype, value.shape, value.stride(), value.storage_offset())
        return traced

    with patch.object(vmap_lowering, "_trace", side_effect=capture):
        lowered = vmap_lowering.lower_vmap_regions(exported.graph_module)
    assert metadata
    for node, expected in metadata.items():
        value = node.meta["val"]
        assert (value.dtype, value.shape, value.stride(), value.storage_offset()) == expected
    converted = convert_model(TorchFXPythonDecoder(lowered))
    compiled = Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    for size in (3, 5):
        _compare(compiled, model, (torch.randn(size, 6, 2),))


def test_export_vmap_existing_attribute():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._ov_vmap_0 = torch.nn.Module()
            self._ov_vmap_0.register_buffer("weight", torch.tensor(2.))

        def forward(self, data):
            return torch.vmap(torch.sin)(data) * self._ov_vmap_0.weight

    model = Model().eval()
    data = torch.randn(3, 4)
    module = torch.export.export(model, (data,)).module()
    decoder = TorchFXPythonDecoder(module)
    assert decoder.pt_module._ov_vmap_0.weight is module._ov_vmap_0.weight
    converted = convert_model(decoder)
    compiled = Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    with torch.no_grad():
        _compare(compiled, model, (data,))


@pytest.mark.parametrize("case", ["same", "different", "mutation", "captured_mutation", "wrapped_mutation"])
def test_export_vmap_rejects(case):
    @torch.no_grad()
    def wrapped_mutation(row):
        return row.clone().add_(1)

    class Model(torch.nn.Module):
        def forward(self, data):
            captured = torch.zeros_like(data[0])

            def captured_mutation(row):
                captured.add_(1)
                return row + captured

            if case in ("same", "different"):
                return torch.vmap(torch.sin, randomness=case)(data)
            functions = {"mutation": lambda row: row.clone().add_(1), "captured_mutation": captured_mutation,
                         "wrapped_mutation": wrapped_mutation}
            return torch.vmap(functions[case])(data)

    match = "randomness='error'" if case in ("same", "different") else "Mutation inside vmap is not supported"
    with pytest.raises(RuntimeError, match=match):
        _convert_export(Model(), (torch.randn(2, 4),))
