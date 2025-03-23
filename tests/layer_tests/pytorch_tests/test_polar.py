# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestPolar(PytorchLayerTest):
    def _prepare_input(self):
        return (
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([0.1, 0.2, 0.3], dtype=np.float32)
        )

    def create_model(self):
        class PolarModel(torch.nn.Module):
            def forward(self, abs, angle):
                real = abs * torch.cos(angle)
                imag = abs * torch.sin(angle)
                return torch.stack([real, imag], dim=-1)
        return PolarModel(), None, None

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_variant", ["static", "dynamic"])
    def test_polar(self, ie_device, precision, ir_version, input_variant):
        atol = 1e-4 if precision == "FP32" else 1e-3
        rtol = 1e-4
        if input_variant == "static":
            input_data = self._prepare_input()
        else:
            static_input = self._prepare_input()
            input_data = (
                np.expand_dims(static_input[0], axis=0),
                np.expand_dims(static_input[1], axis=0)
            )
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   input_data=input_data, model_trace=True, atol=atol, rtol=rtol)
