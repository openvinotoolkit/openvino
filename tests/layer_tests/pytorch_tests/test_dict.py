# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import torch
from typing import Dict

from pytorch_layer_test_class import PytorchLayerTest


class TestDict(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32),)

    def create_model(self):
        class aten_dict(torch.nn.Module):
            def forward(self, x):
                return {"b": x, "a": x + x, "c": 2 * x}, x / 2

        return aten_dict(), None, "prim::DictConstruct"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dict(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, use_convert_model=True)


class aten_dict_with_types(torch.nn.Module):
    def forward(self, x_dict: Dict[str, torch.Tensor]):
        return x_dict["x1"].to(torch.float32) + x_dict["x2"].to(torch.float32)


class aten_dict_no_types(torch.nn.Module):
    def forward(self, x_dict: Dict[str, torch.Tensor]):
        return x_dict["x1"] + x_dict["x2"]


class TestDictParam(PytorchLayerTest):

    def _prepare_input(self):
        return ({"x1": np.random.randn(2, 5, 3, 4).astype(np.float32),
                "x2": np.random.randn(2, 5, 3, 4).astype(np.float32)},)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform.system() == 'Darwin' and platform.machine() in ('x86_64', 'AMD64'),
                        reason='Ticket - 142190')
    def test_dict_param(self, ie_device, precision, ir_version):
        self._test(aten_dict_with_types(), None, "aten::__getitem__", ie_device, precision,
                   ir_version, trace_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Ticket - 142190')
    def test_dict_param_convert_model(self, ie_device, precision, ir_version):
        self._test(aten_dict_with_types(), None, "aten::__getitem__", ie_device, precision,
                   ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.skipif(platform.system() == 'Darwin', reason='Ticket - 142190')
    def test_dict_param_no_types(self, ie_device, precision, ir_version):
        self._test(aten_dict_no_types(), None, "aten::__getitem__", ie_device, precision,
                   ir_version, trace_model=True, freeze_model=False)
