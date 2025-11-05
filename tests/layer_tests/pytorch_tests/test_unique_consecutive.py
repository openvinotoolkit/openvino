# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_unique_consecutive(torch.nn.Module):
    def __init__(self, dim=None, return_inverse=False, return_counts=False, dtype=None):
        super().__init__()
        self.dim = dim
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.dtype = dtype

    def forward(self, x):
        return torch.unique_consecutive(x, dim=self.dim,
                                        return_inverse=self.return_inverse,
                                        return_counts=self.return_counts,
                                        dtype=self.dtype)
class TestUniqueConsecutive(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        size = np.prod(shape)
        data = np.arange(size, dtype=dtype)
        # Shuffle data to have non-consecutive duplicates
        np.random.shuffle(data)
        # Sort data to create consecutive duplicates
        data = np.sort(data)
        data = data.reshape(shape)
        return (data,)

    @pytest.mark.parametrize("shape, dim, return_inverse, return_counts, dtype", [
        ([10], None, False, False, None),
        ([10], None, True, False, None),
        ([10], None, True, True, None),
        ([3, 4], 1, False, False, None),
        ([3, 4], 0, True, False, None),
        ([3, 4], -1, True, True, None),
        ([2, 3, 4], 1, False, False, torch.int64),
        ([2, 3, 4], 2, True, True, torch.int32),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unique_consecutive(self, shape, dim, return_inverse, return_counts, dtype,
                                ie_device, precision, ir_version):
        self._test(aten_unique_consecutive(dim, return_inverse, return_counts, dtype),
                   None,
                   "aten::unique_consecutive",
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"shape": shape, "dtype": np.int32},
                   trace_model=True,
                   )