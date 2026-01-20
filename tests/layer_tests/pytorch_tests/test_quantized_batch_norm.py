# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class quantized_batch_norm2d(torch.nn.Module):
    def __init__(self, num_features, eps, scale, zero_point, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype
        self.eps = eps
        self.num_features = num_features
        # Initialize batch norm parameters
        self.weight = torch.ones(num_features)
        self.bias = torch.zeros(num_features)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, input_tensor):
        quantized_tensor = torch.quantize_per_tensor(input_tensor, 1.0, 0, self.dtype)
        q_batch_norm = torch.ops.quantized.batch_norm2d(
            quantized_tensor,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            self.eps,
            self.scale,
            self.zero_point
        )
        dequantized_tensor = torch.dequantize(q_batch_norm)
        return dequantized_tensor


class TestQuantizedBatchNorm2d(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self, shape):
        return (np.round(5.00 * self.rng.random(shape, dtype=np.float32) - 2.50, 4),)

    @pytest.mark.parametrize("shape", [
        [1, 3, 4, 4],
        [2, 4, 6, 6],
        [1, 16, 8, 8],
    ])
    @pytest.mark.parametrize("eps", [
        1e-5, 1e-3
    ])
    @pytest.mark.parametrize("scale", [
        1.0, 0.21, 0.62
    ])
    @pytest.mark.parametrize("zero_point", [
        0, 4
    ])
    @pytest.mark.parametrize("dtype", [
        torch.quint8,
        torch.qint8
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_batch_norm2d(self, shape, eps, scale, zero_point, dtype, ie_device, precision, ir_version):
        if dtype == torch.quint8:
            zero_point = abs(zero_point)
        num_features = shape[1]
        self._test(quantized_batch_norm2d(num_features, eps, scale, zero_point, dtype), None, ["quantized::batch_norm2d"],
                   ie_device, precision, ir_version, quantized_ops=True, quant_size=scale,
                   kwargs_to_prepare_input={"shape": shape})
