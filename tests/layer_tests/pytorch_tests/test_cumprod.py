# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}


class aten_cumprod(torch.nn.Module):
    def __init__(self, axis, dtype=None, input_dtype=torch.float32):
        super().__init__()
        self.axis = axis
        self.dtype = dtype
        self.input_dtype = input_dtype

    def forward(self, x):
        x = x.to(self.input_dtype)
        if self.dtype is None:
            return torch.cumprod(x, self.axis)
        return torch.cumprod(x, self.axis, dtype=self.dtype)


class aten_cumprod_out(torch.nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x, y):
        return y, torch.cumprod(x, self.axis, out=y)


class TestCumProd(PytorchLayerTest):
    def _prepare_input(self, input_dtype="float32", shape=(2, 5, 7, 13), out=False):
        if input_dtype == "bool":
            x = self.random.randint(0, 2, size=shape).astype(np.float32)
        elif input_dtype in ("int32", "int64"):
            x = self.random.randint(-3, 4, size=shape).astype(np.float32)
        else:
            x = (self.random.rand(*shape) * 3 - 1.5).astype(input_dtype)
        if out:
            return (x, np.zeros(shape, dtype=x.dtype))
        return (x,)

    @pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -4])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32", "int64", "bool"])
    @pytest.mark.parametrize("dtype", [None, "float32", "int64"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_cumprod(self, axis, input_dtype, dtype, ie_device, precision, ir_version):
        self._test(aten_cumprod(axis, TORCH_DTYPES.get(dtype), TORCH_DTYPES[input_dtype]), "aten::cumprod",
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_dtype": input_dtype})

    @pytest.mark.parametrize("shape", [(1,), (2,), (8,), (16,), (3, 1), (3, 32), (4, 33)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_cumprod_lengths(self, shape, ie_device, precision, ir_version):
        self._test(aten_cumprod(-1), "aten::cumprod", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"shape": shape})

    @pytest.mark.parametrize("axis", [0, -1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cumprod_out(self, axis, ie_device, precision, ir_version):
        self._test(aten_cumprod_out(axis), "aten::cumprod", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"out": True})
