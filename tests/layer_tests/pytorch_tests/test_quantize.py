# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

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

class aten_quantize_per_channel_aten_dequantize(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype, axis) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype
        self.axis = axis

    def forward(self, input_tensor):
        quantized_tensor =  torch.quantize_per_channel(input_tensor, self.scale, self.zero_point, self.axis, self.dtype)
        dequantized_tensor = torch.dequantize(quantized_tensor)
        return dequantized_tensor

class TestQuantizePerTensorDequantize(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(5.00 * np.random.rand(100, 100) + 5.00, dtype=np.float32),)

    @pytest.mark.parametrize("scale", [ 
        1.0, 0.21, 0.62
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8, torch.qint8, # torch.qint32 - Not supported with FakeQuantize
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantize_per_tensor_dequantize(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(aten_quantize_per_tensor_aten_dequantize(scale, zero_point, dtype), None, ["aten::quantize_per_tensor", "aten::dequantize"], 
                ie_device, precision, ir_version, )

class TestQuantizePerChannelDequantize(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(5.00 * np.random.rand(5, 6, 7, 8) + 5.00, dtype=np.float32),)

    @pytest.mark.parametrize("scale", [ 
        1.0, 0.21, 0.62
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8, torch.qint8, # torch.qint32 - Not supported with FakeQuantize
    ])
    @pytest.mark.parametrize("axis", [
        0, 1, 2, 3
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantize_per_tensor_dequantize(self, scale, zero_point, dtype, axis, ie_device, precision, ir_version):
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(aten_quantize_per_channel_aten_dequantize(scale, zero_point, dtype, axis), None, ["aten::quantize_per_channel", "aten::dequantize"], 
                ie_device, precision, ir_version, )
