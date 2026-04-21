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
