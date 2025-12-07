# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform
import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class quantized_linear_relu(torch.nn.Module):
    def __init__(self, weight_shape, bias, scale, zero_point, dtype) -> None:
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)
        self.dtype = dtype

        out_features = weight_shape[0] if len(weight_shape) > 1 else 1
        in_features = weight_shape[-1]

        # Initialize float weights and bias (device agnostic)
        self.W = torch.randn(out_features, in_features, dtype=torch.float32)
        self.b = torch.randn(out_features, dtype=torch.float32) if bias else None

        # Prepack weight (quantized operator requires prepacked weights)
        self.W_prepack = torch.ops.quantized.linear_prepack(self.W, self.b)

    def forward(self, input_tensor):
        # Quantize input on the same device as the tensor
        quantized_input = torch.quantize_per_tensor(
            input_tensor, scale=1.0, zero_point=0, dtype=self.dtype
        )

        # Run quantized linear+ReLU
        q_out = torch.ops.quantized.linear_relu(
            quantized_input, self.W_prepack, self.scale, self.zero_point
        )

        # Dequantize output
        return torch.dequantize(q_out)


class TestQuantizedLinearReLU(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        # Return a numpy array (can be converted to any device later)
        return (np.round(5.0 * self.rng.random([3, 9], dtype=np.float32) - 2.5, 4),)

    @pytest.mark.parametrize("weight_shape,bias", [
        ([10, 9], False),
        ([10, 9], True),
        ([9], False),
        ([9], True)
    ])
    @pytest.mark.parametrize("scale", [
        1.0, 0.3, 1.3
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 1
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8,
        torch.qint8
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_linear_relu(self, weight_shape, bias, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8:
            zero_point = abs(zero_point)

        # Run test on requested device (ie_device) instead of forcing CPU
        self._test(
            quantized_linear_relu(weight_shape, bias, scale, zero_point, dtype),
            None,
            ["quantized::linear_relu"],
            ie_device,  # use requested device
            precision,
            ir_version,
            quantized_ops=True,
            quant_size=scale
        )
