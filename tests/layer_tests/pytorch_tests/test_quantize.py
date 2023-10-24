# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

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
    def __init__(self, scales, zero_points, dtype, axis) -> None:
        torch.nn.Module.__init__(self)
        self.scales = torch.Tensor(scales)
        self.zero_points = torch.Tensor(zero_points)
        self.dtype = dtype
        self.axis = axis
    def forward(self, input_tensor):
        quantized_tensor =  torch.quantize_per_channel(input_tensor, self.scales, self.zero_points, self.axis, self.dtype)
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
        torch.quint8,
        torch.qint8,
        pytest.param(torch.qint32, marks=pytest.mark.skip(
            reason="Not supported with FakeQuantize."))
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantize_per_tensor_dequantize(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(aten_quantize_per_tensor_aten_dequantize(scale, zero_point, dtype), None, ["aten::quantize_per_tensor", "aten::dequantize"], 
                ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)

class TestQuantizePerChannelDequantize(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(5.00 * np.random.rand(5, 6, 7, 8) + 5.00, dtype=np.float32),)

    @pytest.mark.parametrize("scale, zero_point, axis", [ 
        [
            np.array([1.0, 0.21, 0.62, 0.5, 0.74], dtype=np.float32),
            np.array([0, -1, 2, -3, 4], dtype=np.int32),
            0
        ],
        [
            np.array([1.0, 0.62, 0.74, 0.11, 0.89, 0.32], dtype=np.float32),
            np.array([0, 2, 4, -5, 6, -7], dtype=np.int32),
            1
        ],
        pytest.param(
            np.array([1.0, 0.21, 0.62, 0.5, 0.11, 0.89, 0.32], dtype=np.float32),
            np.array([0, -1, 2, -3, 4, -5, -7], dtype=np.int32),
            2, 
            marks=pytest.mark.skip(reason="Axis = 2 not supported in FakeQuantize.")),
        [
            np.array([1.0, 0.21, 0.62, 0.5, 0.74, 0.11, 0.89, 0.32], dtype=np.float32),
            np.array([0, -1, 2, -3, 4, -5, 6, -7], dtype=np.int32),
            3
        ],
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8,
        torch.qint8,
        pytest.param(torch.qint32, marks=pytest.mark.skip(
            reason="Not supported with FakeQuantize."))
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantize_per_channel_dequantize(self, scale, zero_point, dtype, axis, ie_device, precision, ir_version):
        np.random.shuffle(scale), np.random.shuffle(zero_point)
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(aten_quantize_per_channel_aten_dequantize(scale, zero_point, dtype, axis), None, ["aten::quantize_per_channel", "aten::dequantize"], 
                ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)
