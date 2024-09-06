# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch
import torch.ao.quantization.fx._decomposed

from pytorch_layer_test_class import PytorchLayerTest


class aten_quantize_per_tensor_aten_dequantize(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor):
        quantized_tensor = torch.quantize_per_tensor(input_tensor, self.scale, self.zero_point, self.dtype)
        dequantized_tensor = torch.dequantize(quantized_tensor)
        return dequantized_tensor


class quantized_decomposed_quantize_per_tensor_aten_dequantize(torch.nn.Module):
    def __init__(self, scale, zero_point, quant_min, quant_max, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = torch.tensor(scale, dtype=torch.float)
        self.zero_point = torch.tensor(zero_point, dtype=torch.float)
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.dtype = dtype

    def forward(self, input_tensor):
        quantized_tensor = torch.ops.quantized_decomposed.quantize_per_tensor(input_tensor, scale=self.scale,
                zero_point=self.zero_point, quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype)
        dequantized_tensor = torch.ops.quantized_decomposed.dequantize_per_tensor(quantized_tensor, scale=self.scale,
                zero_point=self.zero_point, quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype)
        return dequantized_tensor


class aten_quantize_per_channel_aten_dequantize(torch.nn.Module):
    def __init__(self, scales, zero_points, dtype, axis) -> None:
        torch.nn.Module.__init__(self)
        self.scales = torch.Tensor(scales)
        self.zero_points = torch.Tensor(zero_points)
        self.dtype = dtype
        self.axis = axis
    def forward(self, input_tensor):
        quantized_tensor = torch.quantize_per_channel(input_tensor, self.scales, self.zero_points, self.axis, self.dtype)
        dequantized_tensor = torch.dequantize(quantized_tensor)
        return dequantized_tensor


class quantized_decomposed_quantize_per_channel_aten_dequantize(torch.nn.Module):
    def __init__(self, scales, zero_points, quant_min, quant_max, dtype, axis) -> None:
        torch.nn.Module.__init__(self)
        self.scales = torch.tensor(scales, dtype=torch.float)
        self.zero_points = torch.tensor(zero_points, dtype=torch.float)
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.dtype = dtype
        self.axis = axis

    def forward(self, input_tensor):
        quantized_tensor = torch.ops.quantized_decomposed.quantize_per_channel(input_tensor, scales=self.scales,
                zero_points=self.zero_points, axis=self.axis, quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype)
        dequantized_tensor = torch.ops.quantized_decomposed.dequantize_per_channel(quantized_tensor, scales=self.scales,
                zero_points=self.zero_points, axis=self.axis, quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype)
        return dequantized_tensor


class TestQuantizePerTensorDequantize(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (5.00 * self.rng.random([100, 100], dtype=np.float32) + 5.00,)

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


class TestDecomposedQuantizePerTensorDequantize(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (5.00 * self.rng.random([100, 100], dtype=np.float32) + 5.00,)

    @pytest.mark.parametrize("scale", [ 
        1.0, 0.21, 0.62
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4, -7
    ])
    @pytest.mark.parametrize("dtype", [
        torch.uint8,
        torch.int8,
    ])
    @pytest.mark.precommit_fx_backend
    def test_decomposed_quantize_per_tensor_dequantize(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        kwargs = {}
        kwargs["custom_eps"] = 0.15
        quant_min = -128
        quant_max = 127
        if dtype == torch.uint8:
            zero_point = abs(zero_point)
            quant_min = 0
            quant_max = 255
        self._test(quantized_decomposed_quantize_per_tensor_aten_dequantize(scale,
                zero_point, quant_min, quant_max, dtype), None, ["aten::quantize_per_tensor", "aten::dequantize"],
                ie_device, precision, ir_version, quantized_ops=True, quant_size=scale, **kwargs)


class TestQuantizePerChannelDequantize(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (5.00 * self.rng.random([5, 6, 7, 8], dtype=np.float32) + 5.00,)

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
        self.rng.shuffle(scale)
        self.rng.shuffle(zero_point)
        if dtype == torch.quint8: zero_point = abs(zero_point)
        self._test(aten_quantize_per_channel_aten_dequantize(scale, zero_point, dtype, axis), None, ["aten::quantize_per_channel", "aten::dequantize"],
                ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)


class TestDecomposedQuantizePerChannelDequantize(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (5.00 * self.rng.random([5, 6, 7, 8], dtype=np.float32) + 5.00,)

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
        torch.uint8,
        torch.int8,
    ])
    @pytest.mark.precommit_fx_backend
    def test_decomposed_quantize_per_channel_dequantize(self, scale, zero_point, dtype, axis, ie_device, precision, ir_version):
        kwargs = {}
        kwargs["custom_eps"] = 0.15
        self.rng.shuffle(scale)
        self.rng.shuffle(zero_point)
        quant_min = -128
        quant_max = 127
        if dtype == torch.uint8:
            zero_point = abs(zero_point)
            quant_min = 0
            quant_max = 255
        self._test(quantized_decomposed_quantize_per_channel_aten_dequantize(scale,
                zero_point, quant_min, quant_max, dtype, axis), None, ["aten::quantize_per_tensor", "aten::dequantize"],
                ie_device, precision, ir_version, quantized_ops=True, quant_size=scale, **kwargs)
