# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class quantized_relu6(torch.nn.Module):
    def __init__(self, scale, zero_point, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype

    def forward(self, input_tensor1):
        quantized_tensor1 = torch.quantize_per_tensor(input_tensor1, self.scale, self.zero_point, self.dtype)
        q_relu6 = torch.ops.quantized.relu6(quantized_tensor1)
        dequantized_tensor = torch.dequantize(q_relu6)
        return dequantized_tensor


class TestQuantizedRelu6(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (np.round(5.00 * self.rng.random([10, 10], dtype=np.float32) - 2.50, 4),)

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
    def test_quantized_relu6(self, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8:
            zero_point = abs(zero_point)
        self._test(quantized_relu6(scale, zero_point, dtype), None, ["quantized::relu6"],
                   ie_device, precision, ir_version, quantized_ops=True, quant_size=scale)
