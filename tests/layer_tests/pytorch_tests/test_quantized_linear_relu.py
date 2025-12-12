# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class QuantizedLinearReLU(torch.nn.Module):
    def __init__(self, weight_shape, bias, scale, zero_point):
        super().__init__()
        self.linear_relu = torch.ao.nn.intrinsic.quantized.LinearReLU(
            weight_shape[-1], weight_shape[0], bias
        )
        if bias:
            torch.nn.init.normal_(self.linear_relu.bias())
        self.linear_relu.scale = float(scale)
        self.linear_relu.zero_point = int(zero_point)

    def forward(self, inp):
        inp_q = torch.quantize_per_tensor(inp, self.linear_relu.scale, self.linear_relu.zero_point, torch.quint8)
        return torch.dequantize(self.linear_relu(inp_q))


class TestQuantizedLinear(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self, input_shape=(2, 2)):
        return (np.round(self.rng.random(input_shape, dtype=np.float32), 4),)

    @pytest.mark.parametrize("params", [
        {'input_shape': [3, 9], 'weight_shape': [10, 9]},

        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9]},
        {'input_shape': [3, 9], 'weight_shape': [9], "bias": True},
        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9], "bias": True},
    ])
    @pytest.mark.parametrize("scale", [1., 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.parametrize("trace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_linear_relu(self, params, scale, zero_point, trace, ie_device, precision, ir_version):
        input_shape = params.get("input_shape")
        weight_shape = params.get("weight_shape")
        bias = params.get("bias", False)

        model = QuantizedLinearReLU(weight_shape, bias, scale, zero_point)
        ref_net = None

        self._test(
            model,
            ref_net,
            ["quantized::linear_relu"],
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape},
            trace_model=trace,
            freeze_model=False,
            quantized_ops=True,
            quant_size=scale
        )
