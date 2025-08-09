# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestGet(PytorchLayerTest):
    def _prepare_input(self):
        # Prepare a simple 2x3 input tensor
        return (np.random.randn(2, 3).astype(np.float32),)

    def create_model(self, idx):
        class aten_get_model(torch.nn.Module):
            def __init__(self, idx):
                super().__init__()
                self.idx = idx

            def forward(self, x):
                max_result = torch.max(x, dim=1)
                return max_result[self.idx]

        ref_net = None
        return aten_get_model(idx), ref_net, None

    @pytest.mark.parametrize("idx", [
        0,  # Test getting the 'values' tensor from torch.max
        1   # Test getting the 'indices' tensor from torch.max
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_get(self, idx, ie_device, precision, ir_version):
        self._test(*self.create_model(idx), ie_device, precision, ir_version)