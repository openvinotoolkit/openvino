# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestAutocastToFullPrecision(PytorchLayerTest):
    def _prepare_input(self):
        # Generate FP16 input to test the conversion up to FP32
        return (np.random.randn(2, 3).astype(np.float16),)

    def create_model(self, cuda_enabled, cpu_enabled):
        class aten_autocast_to_full_precision(torch.nn.Module):
            def __init__(self, cuda_enabled, cpu_enabled):
                super(aten_autocast_to_full_precision, self).__init__()
                self.cuda_enabled = cuda_enabled
                self.cpu_enabled = cpu_enabled

            def forward(self, x):
                # Call the specific ATen operator
                return torch.ops.aten._autocast_to_full_precision(
                    x, self.cuda_enabled, self.cpu_enabled
                )

        return aten_autocast_to_full_precision(cuda_enabled, cpu_enabled), None, "aten::_autocast_to_full_precision"

    @pytest.mark.parametrize("cuda_enabled", [True, False])
    @pytest.mark.parametrize("cpu_enabled", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_autocast_to_full_precision(self, ie_device, precision, ir_version, cuda_enabled, cpu_enabled):
        self._test(*self.create_model(cuda_enabled, cpu_enabled), 
                   ie_device, precision, ir_version)
