# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class quantized_hardswish(torch.nn.Module):
    def __init__(self, scale, zero_point) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point

    def forward(self, input_tensor1, input_tensor2):
        quantized_tensor1 =  torch.quantize_per_tensor(input_tensor1, self.scale, self.zero_point, torch.qint8)
        quantized_tensor2 =  torch.quantize_per_tensor(input_tensor2, self.scale, self.zero_point, torch.qint8)
        quantized_hardswish = torch.hardswish(quantized_tensor1 + quantized_tensor2)
        dequantized_tensor = torch.dequantize(quantized_hardswish)
        return dequantized_tensor

class TestQuantizedHardswish(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(5.00 * np.random.randn(4, 4) + 5.00, dtype=np.float32),
                np.array(5.00 * np.random.randn(4, 4) + 5.00, dtype=np.float32)) # N(5,5)

    @pytest.mark.parametrize("scale", [
        1.0, 0.21, 0.62
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_hardswish(self, scale, zero_point, ie_device, precision, ir_version):
        self._test(quantized_hardswish(scale, zero_point), None, ["quantized::hardswish"], 
                ie_device, precision, ir_version, )
