# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class Model(torch.nn.Module):
    def forward(self, x):
        return x.data


class ModelGrad(torch.nn.Module):
    def forward(self, x):
        y = x * 2.5
        return y.data


class TestPrimData(PytorchLayerTest):
    def _prepare_input(self):
        if self.dtype in (torch.complex64, torch.complex128):
            real = self.random.randn(*self.shape) * 10
            imag = self.random.randn(*self.shape) * 10
            data = real + 1j * imag
            data = data.astype(np.complex128 if self.dtype == torch.complex128 else np.complex64)
        else:
            data = self.random.randn(*self.shape) * 10
        tensor = torch.from_numpy(data).to(self.dtype)
        return (tensor.numpy(),)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int32, torch.int64])
    @pytest.mark.parametrize("shape", [[2, 3, 4], [1, 5], [10]])
    def test_data_basic(self, shape, dtype, ie_device, precision, ir_version):
        self.shape = shape
        self.dtype = dtype
        self._test(Model(), "prim::data", ie_device, precision, ir_version)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
    def test_data_requires_grad(self, dtype, ie_device, precision, ir_version):
        self.shape = (3, 2)
        self.dtype = dtype
        self._test(ModelGrad(), "prim::data", ie_device, precision, ir_version)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    @pytest.mark.parametrize("shape", [[2, 3], [1, 5], [4]])
    @pytest.mark.xfail(
        reason="OpenVINO frontend does not yet support complex tensor inputs",
        raises=AssertionError,
    )
    def test_data_complex(self, shape, dtype, ie_device, precision, ir_version):
        self.shape = shape
        self.dtype = dtype
        self._test(Model(), "prim::data", ie_device, precision, ir_version)

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    @pytest.mark.xfail(
        reason="OpenVINO frontend does not yet support complex tensor inputs",
        raises=AssertionError,
    )
    def test_data_complex_requires_grad(self, dtype, ie_device, precision, ir_version):
        self.shape = (2, 3)
        self.dtype = dtype
        self._test(ModelGrad(), "prim::data", ie_device, precision, ir_version)
