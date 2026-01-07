# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestQuantizedLinearDynamic(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self, input_shape=(2, 2)):
        return (np.round(self.rng.random(input_shape, dtype=np.float32), 4),)

    def create_model(self, weight_shape, is_bias):

        class aten_quantized_linear_dynamic(torch.nn.Module):
            def __init__(self, weight_shape, is_bias):
                super(aten_quantized_linear_dynamic, self).__init__()
                self.linear = torch.ao.nn.quantized.dynamic.Linear(
                    weight_shape[-1], weight_shape[0], is_bias)

            def forward(self, inp):
                return self.linear(inp)

        ref_net = None

        return aten_quantized_linear_dynamic(weight_shape, is_bias), ref_net, "quantized::linear_dynamic"

    @pytest.mark.parametrize("params", [
        {'input_shape': [3, 9], 'weight_shape': [10, 9]},
        {'input_shape': [3, 9], 'weight_shape': [9]},
        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9]},
        {'input_shape': [2, 3, 9], 'weight_shape': [9]},
        {'input_shape': [3, 9], 'weight_shape': [9], "bias": True},
        {'input_shape': [3, 9], 'weight_shape': [10, 9], "bias": True},
        {'input_shape': [2, 3, 9], 'weight_shape': [10, 9], "bias": True},
    ])
    @pytest.mark.parametrize("trace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_linear_dynamic(self, params, trace, ie_device, precision, ir_version):
        input_shape = params.get("input_shape")
        weight_shape = params.get("weight_shape")
        bias = params.get("bias", False)
        self._test(*self.create_model(weight_shape, bias), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape}, trace_model=trace, freeze_model=False)
