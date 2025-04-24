# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class quantized_mul(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor1, input_tensor2):
        quantized_tensor1 = torch.quantize_per_tensor(input_tensor1, 1.0, 0, self.dtype)
        quantized_tensor2 = torch.quantize_per_tensor(input_tensor2, 1.0, 0, self.dtype)
        q_mul = torch.ops.quantized.mul(quantized_tensor1, quantized_tensor2, self.scale, self.zero_point)
        dequantized_tensor = torch.dequantize(q_mul)
        return dequantized_tensor


class TestQuantizedMul(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (np.round(5.00 * self.rng.random([10, 10], dtype=np.float32) - 2.50, 4),
                np.round(5.00 * self.rng.random([10, 10], dtype=np.float32) - 2.50, 4))

    @pytest.mark.parametrize("scale", [
        1.0, 0.21, 0.62, 0.9999
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8,
        torch.qint8
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_mul(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8:
            zero_point = abs(zero_point)
        self._test(quantized_mul(scale, zero_point, dtype), None, ["quantized::mul"],
                   ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)
