# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import random

from pytorch_layer_test_class import PytorchLayerTest


class aten_prod(torch.nn.Module):
    def forward(self, x):
        return torch.prod(x)


class aten_prod_dtype(torch.nn.Module):
    def __init__(self, dtype):
        super(aten_prod_dtype, self).__init__()
        self.dtype = dtype

    def forward(self, x):
        return torch.prod(x)


class aten_prod_dim(torch.nn.Module):
    def __init__(self, dim, keepdims):
        super(aten_prod_dim, self).__init__()
        self.dim = dim
        self.keepdims = keepdims

    def forward(self, x):
        return torch.prod(x, self.dim, self.keepdims)


class aten_prod_dim_dtype(torch.nn.Module):
    def __init__(self, dim, keepdims, dtype):
        super(aten_prod_dim_dtype, self).__init__()
        self.dim = dim
        self.keepdims = keepdims
        self.dtype = dtype

    def forward(self, x):
        return torch.prod(x, self.dim, self.keepdims, dtype=self.dtype)


class TestProd(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2)):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    @pytest.mark.parametrize("shape", [(1,),
                                       (2,),
                                       (2, 3),
                                       (3, 4, 5),
                                       (1, 2, 3, 4),
                                       (1, 2, 3, 4, 5)])
    @pytest.mark.parametrize("dtype", [None, torch.int32])
    @pytest.mark.parametrize("has_dim,keepdims", [(False, None), (True, True), (True, False)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_prod(self, ie_device, precision, ir_version, shape, dtype, has_dim, keepdims):
        if dtype is not None:
            if has_dim:
                m = aten_prod_dim_dtype(random.randint(0, len(shape) - 1),
                                        keepdims,
                                        dtype)
            else:
                m = aten_prod_dtype(dtype)
        else:
            if has_dim:
                m = aten_prod_dim(random.randint(0, len(shape) - 1), keepdims)
            else:
                m = aten_prod()
        self._test(m, None, 'aten::prod', ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_shape': shape})
