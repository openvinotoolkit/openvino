# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class quantized_hardswish(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor1):
        quantized_tensor1 =  torch.quantize_per_tensor(input_tensor1, 1.0, 0, self.dtype)
        quantized_hardswish = torch.ops.quantized.hardswish(quantized_tensor1, self.scale, self.zero_point)
        dequantized_tensor = torch.dequantize(quantized_hardswish)
        return dequantized_tensor

class TestQuantizedHardswish(PytorchLayerTest):
    def _prepare_input(self):
        return (np.round(np.array(5.00 * np.random.rand(10, 10) - 2.50, dtype=np.float32), 4),)

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
    def test_quantized_hardswish(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(quantized_hardswish(scale, zero_point, dtype), None, ["quantized::hardswish"], 
                ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)
