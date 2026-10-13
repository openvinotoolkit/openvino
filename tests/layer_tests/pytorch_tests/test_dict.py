# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestDict(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(2, 5, 3, 4),)

    def create_model(self):
        class aten_dict(torch.nn.Module):
            def forward(self, x):
                return {"b": x, "a": x + x, "c": 2 * x}, x / 2

        return aten_dict(), "prim::DictConstruct"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dict(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, use_convert_model=True)


class aten_dict_with_types(torch.nn.Module):
    def forward(self, x_dict: dict[str, torch.Tensor]):
        return x_dict["x1"].to(torch.float32) + x_dict["x2"].to(torch.float32)


class aten_dict_no_types(torch.nn.Module):
    def forward(self, x_dict: dict[str, torch.Tensor]):
        return x_dict["x1"] + x_dict["x2"]


class TestDictParam(PytorchLayerTest):

    def _prepare_input(self):
        return ({"x1": self.random.randn(2, 5, 3, 4),
            "x2": self.random.randn(2, 5, 3, 4)},)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform.system() == 'Darwin' and platform.machine() in ('x86_64', 'AMD64'),
                        reason='Ticket - 142190')
    def test_dict_param(self, ie_device, precision, ir_version):
        self._test(aten_dict_with_types(), "aten::__getitem__", ie_device, precision,
                   ir_version, trace_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Ticket - 142190')
    def test_dict_param_convert_model(self, ie_device, precision, ir_version):
        self._test(aten_dict_with_types(), "aten::__getitem__", ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Ticket - 142190')
    def test_dict_param_no_types(self, ie_device, precision, ir_version):
        self._test(aten_dict_no_types(), "aten::__getitem__", ie_device, precision,
                   ir_version, trace_model=True, freeze_model=False)


class aten_dict_mixed_inputs(torch.nn.Module):
    def forward(self, x_dict: dict[str, torch.Tensor], y: torch.Tensor):
        # one dict key is consumed alongside a regular tensor input; the other bypasses
        return x_dict["x1"] + y, x_dict["x2"]


class TestDictParamMixed(PytorchLayerTest):

    def _prepare_input(self):
        x1 = self.random.randn(1, 3, 4)
        x2 = self.random.randn(1, 3, 4)
        y = self.random.randn(1, 3, 4)
        return ({"x1": x1, "x2": x2}, y)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dict_param_mixed_inputs(self, ie_device, precision, ir_version):
        # Regression: ensure dict parameter resolution does not drop unrelated inputs
        self._test(aten_dict_mixed_inputs(), "aten::__getitem__", ie_device, precision,
                   ir_version, trace_model=True, freeze_model=False)


class dict_out_single(torch.nn.Module):
    def forward(self, x):
        return {"out": x.relu()}


class dict_out_nested(torch.nn.Module):
    def forward(self, x):
        return {"a": x + 1, "inner": {"b": x * 2, "c": x - 1}}, x / 2


class dict_out_duplicated(torch.nn.Module):
    def forward(self, x):
        y = x + 1
        return {"a": y, "b": y, "c": x * 3}


class dict_out_same_nested_key(torch.nn.Module):
    def forward(self, x):
        return {"p": {"k": x + 1}, "q": {"k": x + 2}}


class dict_out_node_name_key(torch.nn.Module):
    def forward(self, x):
        # "mul" and "x" are also names of graph nodes
        return {"mul": x * 2 + 1, "x": x - 1}


@pytest.mark.nightly
@pytest.mark.precommit_torch_export
@pytest.mark.parametrize("model,keys,expected_names", [
    (dict_out_single, {"out"}, ["out"]),
    (dict_out_nested, {"a", "inner", "b", "c"}, ["a", "b", "c", None]),
    # one tensor under several keys can't get a single name
    (dict_out_duplicated, {"a", "b", "c"}, [None, None, "c"]),
    (dict_out_same_nested_key, {"p", "q", "k"}, [None, None]),
    (dict_out_node_name_key, {"mul", "x"}, [None, None]),
], ids=["single", "nested", "duplicated", "same_nested_key", "node_name_key"])
def test_dict_output_names_export(model, keys, expected_names, ie_device, precision):
    import numpy as np
    from openvino import compile_model, convert_model
    from torch.utils._pytree import tree_leaves

    x = torch.randn(2, 5, 3, 4)
    m = model()
    ov_model = convert_model(torch.export.export(m, (x,)))
    fw_res = tree_leaves(m(x))
    assert len(ov_model.outputs) == len(expected_names)
    for ov_out, expected in zip(ov_model.outputs, expected_names):
        names = ov_out.get_names()
        if expected is None:
            assert not names & keys, f"unexpected dict key in output names: {names}"
        else:
            assert expected in names, names
    compiled = compile_model(ov_model, ie_device, {"INFERENCE_PRECISION_HINT": "f32"})
    ov_res = compiled(x.numpy())
    for i, expected in enumerate(expected_names):
        out = compiled.output(expected) if expected else compiled.output(i)
        np.testing.assert_allclose(ov_res[out], fw_res[i].numpy(), rtol=1e-5, atol=1e-5)
