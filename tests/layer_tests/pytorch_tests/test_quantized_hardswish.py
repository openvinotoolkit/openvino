# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
        self.hswish = torch.nn.Hardswish()

    def forward(self, input_tensor1):
        quantized_tensor1 =  torch.quantize_per_tensor(input_tensor1, self.scale, self.zero_point, self.dtype)
        # dequantized_tensor = torch.dequantize(quantized_tensor1)
        # quantized_hardswish = self.hswish(dequantized_tensor)
        # quantized_tensor1 =  torch.quantize_per_tensor(quantized_hardswish, self.scale, self.zero_point, self.dtype)
        # dequantized_tensor = torch.dequantize(quantized_tensor1)
        # return dequantized_tensor 
        quantized_hardswish = torch.ops.quantized.hardswish(quantized_tensor1, self.scale, self.zero_point)
        dequantized_tensor = torch.dequantize(quantized_hardswish)
        return dequantized_tensor

class TestQuantizedHardswish(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(5.00 * np.random.randn(100, 100) + 5.00, dtype=np.float32),)

    @pytest.mark.parametrize("scale", [
        # 1.0, 
        0.21, 
        # 0.62,
    ])
    @pytest.mark.parametrize("zero_point", [
        # 0, 4, 
        -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8, torch.qint8, # torch.qint32
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_hardswish(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(quantized_hardswish(scale, zero_point, dtype), None, ["aten::dequantize"],# ["quantized::hardswish"], 
                ie_device, precision, ir_version, )
