# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_prod(torch.nn.Module):
    def __init__(self, in_dtype):
        super().__init__()
        self.in_dtype = in_dtype

    def forward(self, x):
        return torch.prod(x.to(self.in_dtype))


class aten_prod_dtype(torch.nn.Module):
    def __init__(self, dtype, in_dtype):
        super().__init__()
        self.dtype = dtype
        self.in_dtype = in_dtype

    def forward(self, x):
        return torch.prod(x.to(self.in_dtype), dtype=self.dtype)


class aten_prod_dim(torch.nn.Module):
    def __init__(self, dim, keepdims, in_dtype):
        super().__init__()
        self.dim = dim
        self.keepdims = keepdims
        self.in_dtype = in_dtype

    def forward(self, x):
        return torch.prod(x.to(self.in_dtype), self.dim, self.keepdims)


class aten_prod_dim_dtype(torch.nn.Module):
    def __init__(self, dim, keepdims, dtype, in_dtype):
        super().__init__()
        self.dim = dim
        self.keepdims = keepdims
        self.dtype = dtype
        self.in_dtype = in_dtype

    def forward(self, x):
        return torch.prod(x.to(self.in_dtype), self.dim, self.keepdims, dtype=self.dtype)


class TestProd(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2), dtype=torch.float32):
        return (self.random.randn(*input_shape, dtype=dtype),)

    @pytest.mark.parametrize("shape", [(1,),
                                       (2,),
                                       (2, 3),
                                       (3, 4, 5),
                                       (1, 2, 3, 4),
                                       (1, 2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", [None, torch.int32])
    @pytest.mark.parametrize("in_dtype", [torch.float32, torch.bool])
    @pytest.mark.parametrize("has_dim,keepdims", [(False, None), (True, True), (True, False)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_prod(self, ie_device, precision, ir_version, shape, dtype, in_dtype, has_dim, keepdims):
        if dtype is not None:
            if has_dim:
                m = aten_prod_dim_dtype(random.randint(0, len(shape) - 1),
                                        keepdims,
                                        dtype,
                                        in_dtype)
            else:
                m = aten_prod_dtype(dtype, in_dtype)
        else:
            if has_dim:
                m = aten_prod_dim(random.randint(0, len(shape) - 1),
                                  keepdims,
                                  in_dtype)
            else:
                m = aten_prod(in_dtype)
        self._test(m, 'aten::prod', ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_shape': shape, 'dtype': in_dtype})


class aten_prod_with_dtype(torch.nn.Module):
    def __init__(self, dim, keepdims, dtype):
        super().__init__()
        self.dim = dim
        self.keepdims = keepdims
        self.dtype = dtype

    def forward(self, x):
        if self.dim is None:
            if self.dtype is None:
                return torch.prod(x)
            return torch.prod(x, dtype=self.dtype)
        if self.dtype is None:
            return torch.prod(x, self.dim, self.keepdims)
        return torch.prod(x, self.dim, self.keepdims, dtype=self.dtype)


class TestProdIntegerDtypePromotion(PytorchLayerTest):
    def _prepare_input(self, input_values, input_dtype):
        dtype = torch.empty((), dtype=input_dtype).numpy().dtype
        return (np.asarray(input_values, dtype=dtype),)

    @pytest.mark.parametrize(
        "input_dtype,input_values,dim,keepdims,dtype",
        [
            (torch.int8, [100, 2], None, False, None),
            (torch.int16, [300, 300], None, False, None),
            (torch.int32, [50_000, 50_000], None, False, None),
            (torch.int32, [2, 3], None, False, None),
            (torch.int32, [], None, False, None),
            (torch.int32, [7], None, False, None),
            (torch.int64, [50_000, 50_000], None, False, None),
            (torch.bool, [True, False, True], None, False, None),
            (torch.int32, [[50_000, 2], [3, 4]], 1, True, None),
            (torch.int32, [50_000, 50_000], None, False, torch.int32),
        ],
    )
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_prod_integer_dtype_promotion(
        self, ie_device, precision, ir_version, input_dtype, input_values, dim, keepdims, dtype
    ):
        model = aten_prod_with_dtype(dim, keepdims, dtype)
        self._test(
            model,
            'aten::prod',
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={'input_values': input_values, 'input_dtype': input_dtype},
        )
