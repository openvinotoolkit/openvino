# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_unique_consecutive(torch.nn.Module):
    def __init__(self, return_inverse=False, return_counts=False, dim=None):
        super().__init__()
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.dim = dim

    def forward(self, x):
        result, inverse, counts = torch.unique_consecutive(x, self.return_inverse, self.return_counts, self.dim)
        output = (result, )
        if self.return_inverse:
            output += (inverse, )
        if self.return_counts:
            output += (counts, )
        return output

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

    @pytest.mark.parametrize("shape", [
        [10], [3, 4], [2, 3, 4]
    ])
    @pytest.mark.parametrize("return_inverse", [False, True])
    @pytest.mark.parametrize("return_counts", [False, True])
    @pytest.mark.parametrize("dim", [None, 0, 1, -1])   # TODO: need to make multiple functions for different ranks to avoid invalid dim
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unique_consecutive(self, shape, return_inverse, return_counts, dim, dtype,
                                ie_device, precision, ir_version):
        self._test(aten_unique_consecutive(return_inverse, return_counts, dim),
                   None,
                   "aten::unique_consecutive",
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"shape": shape, "dtype": dtype},
                #    trace_model=True,
                   )