# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestQuantizedLinear(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self, input_shape=(2, 2)):
        return (np.round(self.rng.random(input_shape, dtype=np.float32), 4),)

    def create_model(self, weight_shape, is_bias, scale, zero_point):

        class aten_quantized_linear(torch.nn.Module):
            def __init__(self, weight_shape, is_bias, scale, zero_point):
                super(aten_quantized_linear, self).__init__()
                if is_bias:
                    self.linear = torch.ao.nn.quantized.Linear(
                        weight_shape[-1], weight_shape[0], True)
                    torch.nn.init.normal_(self.linear.bias())
                else:
                    self.linear = torch.ao.nn.quantized.Linear(
                        weight_shape[-1], weight_shape[0], False)
                self.linear.scale = float(scale)
                self.linear.zero_point = int(zero_point)

            def forward(self, inp):
                inp_q = torch.quantize_per_tensor(inp, 1.0, 0, torch.quint8)
                return torch.dequantize(self.linear(inp_q))

        ref_net = None

        return aten_quantized_linear(weight_shape, is_bias, scale, zero_point), ref_net, "quantized::linear"

    def create_hardtanh_model(self, weight_shape, is_bias, scale, zero_point, inplace):

        class aten_quantized_linear(torch.nn.Module):
            def __init__(self, weight_shape, is_bias, scale, zero_point, inplace):
                super(aten_quantized_linear, self).__init__()
                self.hardtanh = torch.nn.Hardtanh(inplace=inplace)
                if is_bias:
                    self.linear = torch.ao.nn.quantized.Linear(
                        weight_shape[-1], weight_shape[0], True)
                    torch.nn.init.normal_(self.linear.bias())
                else:
                    self.linear = torch.ao.nn.quantized.Linear(
                        weight_shape[-1], weight_shape[0], False)
                self.linear.scale = float(scale)
                self.linear.zero_point = int(zero_point)

            def forward(self, inp):
                inp_q = torch.quantize_per_tensor(inp, 1., 0, torch.quint8)
                inp_q = self.hardtanh(inp_q)
                return torch.dequantize(self.linear(inp_q))

        return aten_quantized_linear(weight_shape, is_bias, scale, zero_point, inplace), None, ["quantized::linear", "aten::hardtanh_" if inplace else "aten::hardtanh"]

    @pytest.mark.parametrize("params", [
        {'input_shape': [3, 9], 'weight_shape': [10, 9]},
        {'input_shape': [3, 9], 'weight_shape': [9]},
        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9]},
        {'input_shape': [2, 3, 9], 'weight_shape': [9]},
        {'input_shape': [3, 9], 'weight_shape': [9], "bias": True},
        {'input_shape': [3, 9], 'weight_shape': [10, 9], "bias": True},
        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9], "bias": True},
    ])
    @pytest.mark.parametrize("scale", [1., 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.parametrize("trace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_linear(self, params, scale, zero_point, trace, ie_device, precision, ir_version):
        input_shape = params.get("input_shape")
        weight_shape = params.get("weight_shape")
        bias = params.get("bias", False)
        self._test(*self.create_model(weight_shape, bias, scale, zero_point), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape}, trace_model=trace, freeze_model=False, quantized_ops=True, quant_size=scale)

    @pytest.mark.parametrize("trace", [True, False])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_hardtanh_linear(self, trace, inplace, ie_device, precision, ir_version):
        self._test(*self.create_hardtanh_model([10, 9], True, 1, 0.3, inplace), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": [2, 3, 9]}, trace_model=trace, freeze_model=False, quantized_ops=True, quant_size=0.3)
