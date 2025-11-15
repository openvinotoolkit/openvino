# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_kthvalue(torch.nn.Module):
    def __init__(self, k, dim, keepdim):
        super().__init__()
        self.k = k
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        kwargs = {}
        if self.dim is not None:
            kwargs["dim"] = self.dim
        if self.keepdim is not None:
            kwargs["keepdim"] = self.keepdim
        return torch.kthvalue(x, self.k, **kwargs)


class TestKthValue(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        size = np.prod(shape)
        data = np.arange(size, dtype=dtype)
        np.random.shuffle(data)
        data = data.reshape(shape)
        return (data,)

    @pytest.mark.parametrize("shape, k, dim", [
        ([2, 3], 2, 1),
        ([4, 5, 6], 3, 0),
        ([4, 5, 6], 4, -2),
        ([2, 3, 4], 3, None),
        ([10], 1, 0),
        ([3, 8, 4], 8, 1),
        ([3, 8, 4, 2, 2, 2], 8, 1),
    ])
    @pytest.mark.parametrize("keepdim", [False, True])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_kthvalue(self, shape, k, dim, keepdim, dtype, ie_device, precision, ir_version):
        self._test(aten_kthvalue(k, dim, keepdim),
                   None,
                   "aten::kthvalue",
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"shape": shape, "dtype": dtype},
                   trace_model=True,
                   )
