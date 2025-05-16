# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class quantized_linear_relu(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        super().__init__()
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor):
        # Quantize input tensor
        quantized_tensor = torch.quantize_per_tensor(input_tensor, self.scale, self.zero_point, self.dtype)
        # Apply quantized ReLU operation
        q_relu = torch.ops.quantized.relu(quantized_tensor)
        # Dequantize for comparison
        return torch.dequantize(q_relu)


class TestQuantizedLinearReLU(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (np.round(5.00 * self.rng.random([10, 10], dtype=np.float32) - 2.50, 4),)

    @pytest.mark.parametrize("scale", [1.0, 0.21, 0.62, 0.9999])
    @pytest.mark.parametrize("zero_point", [0, 4, -7])
    @pytest.mark.parametrize("dtype", [torch.quint8, torch.qint8])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_linear_relu(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8:
            zero_point = abs(zero_point)  # Ensure valid zero_point for unsigned int
        self._test(quantized_linear_relu(scale, zero_point, dtype), None, ["quantized::linear_relu"],
                   ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)
