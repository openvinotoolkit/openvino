# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestQuantizedLinear(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2)):
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, weight_shape, is_bias):

        class aten_quantized_linear(torch.nn.Module):
            def __init__(self, weight_shape, is_bias):
                super(aten_quantized_linear, self).__init__()
                if is_bias:
                    self.linear = torch.ao.nn.quantized.Linear(weight_shape[-1], weight_shape[0], True)
                    torch.nn.init.normal_(self.linear.bias())
                else:
                    self.linear = torch.ao.nn.quantized.Linear(weight_shape[-1], weight_shape[0], False)

            def forward(self, inp):
                inp_q = torch.quantize_per_tensor(inp, 0.5, 1, torch.quint8)
                return torch.dequantize(self.linear(inp_q))

        ref_net = None

        return aten_quantized_linear(weight_shape, is_bias), ref_net, "quantized::linear"

    @pytest.mark.parametrize("params", [
        {'input_shape': [3, 9], 'weight_shape': [10, 9]},
        {'input_shape': [3, 9], 'weight_shape': [9]},
        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9]},
        {'input_shape': [2, 3, 9], 'weight_shape': [9]},
        {'input_shape': [3, 9], 'weight_shape': [10, 9], "bias": True},
        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9], "bias": True},
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_linear(self, params, ie_device, precision, ir_version):
        input_shape = params.get("input_shape")
        weight_shape = params.get("weight_shape")
        bias = params.get("bias", False)
        self._test(*self.create_model(weight_shape, bias), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape}, trace_model=True, freeze_model=False)
