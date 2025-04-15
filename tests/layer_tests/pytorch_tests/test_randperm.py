# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestSortedRandperm(PytorchLayerTest):
    def _prepare_input(self):
        return (np.arange(self.n, dtype=np.int64),)

    def create_model(self, n, num_inputs, dtype_value=None):
        class AtenSortedRandperm(torch.nn.Module):
            def __init__(self, n, num_inputs, dtype_value):
                super().__init__()
                self.n = n
                self.num_inputs = num_inputs
                self.dtype = torch.int64 if dtype_value == 4 else None

            def forward(self, x):
                if self.num_inputs == 1:
                    p = torch.randperm(self.n)
                elif self.num_inputs == 2:
                    p = torch.randperm(self.n, dtype=self.dtype)
                elif self.num_inputs == 5:
                    p = torch.randperm(self.n, dtype=self.dtype, layout=torch.strided, 
                                         device=x.device, pin_memory=False)
                else:
                    raise ValueError("Invalid num_inputs")
                # sort to get a deterministic order for verifying the permutation.
                x_permuted = x[p]
                sorted_tensor, _ = torch.sort(x_permuted)
                return sorted_tensor

        return AtenSortedRandperm(n, num_inputs, dtype_value), None, "aten::randperm"

    @pytest.mark.parametrize(("n", "num_inputs", "dtype_value"), [
        (0, 1, None),
        (1, 1, None),
        (5, 1, None),
        (5, 2, 4),
        (5, 5, 4),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sorted_randperm(self, n, num_inputs, dtype_value, ie_device, precision, ir_version):
        self.n = n
        self._test(*self.create_model(n, num_inputs, dtype_value), ie_device, precision, ir_version)
