# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torch import dtype

from pytorch_layer_test_class import PytorchLayerTest

class aten_quantize_per_tensor_aten_dequantize(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor):
        quantized_tensor =  torch.quantize_per_tensor(input_tensor, self.scale, self.zero_point, self.dtype)
        dequantized_tensor = torch.dequantize(quantized_tensor)
        return dequantized_tensor

class TestQuantizePerTensorDequantize(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(100.00 * np.random.randn(5, 9, 7), dtype=np.float32),) # N(0,100)

    @pytest.mark.parametrize("scale", [
        0.1, 0.21, 0.62
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.qint32,
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantize_per_tensor_dequantize(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        self._test(aten_quantize_per_tensor_aten_dequantize(scale, zero_point, dtype), None, ["aten::quantize_per_tensor", "aten::dequantize"], 
                ie_device, precision, ir_version)
