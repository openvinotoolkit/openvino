# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class aten_reverse(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.flip(x, self.dims)


class TestReverse(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        size = np.prod(shape)
        data = np.arange(size, dtype=dtype)
        np.random.shuffle(data)
        data = data.reshape(shape)
        return (data,)

    @pytest.mark.parametrize("shape, dims", [
        ([4, 5, 6], [0]),
        ([4, 5, 6], [1, 2]),
        ([10, 10], [-1]),
        ([2, 3, 4, 5], [0, 2]),
        # 1D Tensor
        ([3], [0]),
        # 2D Tensor
        ([2, 2], [0]),
        ([2, 2], [1]),
        ([2, 2], [0, 1]),
        # 3D Tensor
        ([2, 2, 2], [0]),
        ([2, 2, 2], [1]),
        ([2, 2, 2], [2]),
    ])
    @pytest.mark.parametrize("dtype", [np.float32, np.int32])
    def test_reverse(self, shape, dims, dtype, ie_device, precision, ir_version):
        self._test(
            aten_reverse(dims),
            None,
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"shape": shape, "dtype": dtype},
            trace_model=True,
        )
