# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestSoftmax(PytorchLayerTest):
    def _prepare_input(self, second_input_dtype=None):
        if second_input_dtype is None:
            return (self.random.randn(1, 3, 224, 224),)
        return (self.random.randn(1, 3, 224, 224), self.random.randn(1, dtype=second_input_dtype))

    def create_model(self, dim, dtype=None, use_prim_dtype=False):
        import torch
        import torch.nn.functional as F
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32
        }
        dtype = dtype_map.get(dtype)

        class aten_softmax(torch.nn.Module):
            def __init__(self, dim, dtype, use_prim_dtype):
                super().__init__()
                self.dim = dim
                self.dtype = dtype
                if use_prim_dtype:
                    self.forward = self.forward_prim_dtype
                elif dtype is not None:
                    self.forward = self.forward_dtype

            def forward(self, x):
                return F.softmax(x, self.dim)

            def forward_dtype(self, x):
                return F.softmax(x, self.dim, dtype=self.dtype)

            def forward_prim_dtype(self, x, y):
                return F.softmax(x, self.dim, dtype=y.dtype)


        return aten_softmax(dim, dtype, use_prim_dtype), "aten::softmax"

    @pytest.mark.parametrize("dim", [-1, 3])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_softmax(self, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim), ie_device, precision, ir_version)

    @pytest.mark.parametrize("dim", [-1, 3])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("use_prim_dtype", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_softmax(self, dim, dtype, use_prim_dtype, ie_device, precision, ir_version):
        input_kwargs = {}
        if use_prim_dtype:
            input_kwargs["second_input_dtype"] = dtype
        self._test(*self.create_model(dim, dtype, use_prim_dtype), ie_device,
                   precision, ir_version, kwargs_to_prepare_input=input_kwargs)


class aten_softmax_scalar(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, self.dim)


class TestSoftmaxScalar(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1).reshape(()).astype(np.float32),)

    @pytest.mark.parametrize("dim", [-1, 0])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_softmax_scalar(self, dim, ie_device, precision, ir_version):
        self._test(aten_softmax_scalar(dim), None, "aten::softmax",
                   ie_device, precision, ir_version)
