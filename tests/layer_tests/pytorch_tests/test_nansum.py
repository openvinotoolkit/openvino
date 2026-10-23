# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestNanSum(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([[np.nan, 1, -2], [np.nan, np.nan, np.nan]], dtype=np.float32),)

    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dim", [None, 0, 1, ()])
    @pytest.mark.parametrize("keepdim", [False, True])
    def test_nansum(self, ie_device, precision, ir_version, dim, keepdim):
        class NanSum(torch.nn.Module):
            def forward(self, value):
                return torch.nansum(value, dim=dim, keepdim=keepdim)

        self._test(NanSum(), "aten::nansum", ie_device, precision, ir_version, trace_model=True)


class TestNanSumDtype(PytorchLayerTest):
    def _prepare_input(self):
        if self.input_dtype == "int32":
            return (np.array([[2**24, 1], [-2**24, 1]], dtype=np.int32),)
        return (np.array([[np.nan, 4, 1, -4], [np.inf, -np.inf, 0, 0]], dtype=np.float32),)

    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("input_dtype,dtype", [("int32", None), ("float32", torch.float64)])
    def test_nansum_dtype(self, ie_device, precision, ir_version, input_dtype, dtype):
        class NanSum(torch.nn.Module):
            def forward(self, value):
                return torch.nansum(value, dim=1, dtype=dtype)

        self.input_dtype = input_dtype
        self._test(NanSum(), "aten::nansum", ie_device, precision, ir_version, trace_model=True)
