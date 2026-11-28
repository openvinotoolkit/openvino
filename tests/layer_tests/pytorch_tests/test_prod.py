# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import random

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
    def _prepare_input(self, input_shape=(2,), dtype=torch.float32):
        shape = tuple(input_shape) if isinstance(input_shape, (list, tuple)) else (input_shape,)
        if dtype in [torch.int8, torch.int16, torch.int32]:
            return (self.random.randint(-1, 2, size=shape, dtype=dtype),)
        elif dtype in [torch.uint8, torch.bool]:
            return (self.random.randint(0, 2, size=shape, dtype=dtype),)
        return (self.random.randn(*shape, dtype=dtype),)

    @pytest.mark.parametrize("shape", [(1,),
                                       (2,),
                                       (2, 3),
                                       (3, 4, 5),
                                       (1, 2, 3, 4),
                                       (1, 2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", [None, torch.int32])
    @pytest.mark.parametrize("in_dtype", [torch.float32, torch.bool, torch.int8, torch.uint8, torch.int16, torch.int32])
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


class aten_prod_overflow(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.prod(x) if self.dim is None else torch.prod(x, self.dim)


class TestProdIntegerOverflow(PytorchLayerTest):
    # Integer prod must be computed in int64 like PyTorch does (issue #38719),
    # including products that do not fit into int32.
    def _prepare_input(self, values, dtype):
        return (np.array(values, dtype=dtype),)

    @pytest.mark.parametrize("values,dtype,dim", [
        ([100, 100], "int8", None),             # 10000: overflows int8
        ([16, 16], "uint8", None),              # 256: overflows uint8
        ([1000, 1000, 1000, 1000], "int32", None),  # 1e12: overflows int32
        ([[70000, 70000], [3, 5]], "int32", 1),     # 4.9e9 along a dim
        ([[70000, 70000], [70000, 5]], "int32", 0),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_prod_integer_overflow(self, values, dtype, dim, ie_device, precision, ir_version):
        self._test(aten_prod_overflow(dim), 'aten::prod', ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'values': values, 'dtype': dtype}, use_convert_model=True)
