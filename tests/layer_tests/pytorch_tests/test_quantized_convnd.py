# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import numpy as np
import torch

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from pytorch_layer_test_class import PytorchLayerTest


class TestQuantizedConv2D(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (np.round(self.rng.random([2, 3, 25, 25], dtype=np.float32), 4),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias, relu, scale, zero_point):
        class quantized_conv2d(torch.nn.Module):
            def __init__(self):
                super(quantized_conv2d, self).__init__()
                if not relu:
                    conv_func = torch.ao.nn.quantized.Conv2d
                else:
                    conv_func = torch.ao.nn.intrinsic.quantized.ConvReLU2d
                self.conv = conv_func(
                    weights_shape[1] * groups,
                    weights_shape[0],
                    weights_shape[2:],
                    strides,
                    pads,
                    dilations,
                    groups,
                    bias,
                )
                if bias:
                    torch.nn.init.normal_(self.conv.bias())
                self.conv.scale = float(scale)
                self.conv.zero_point = int(zero_point)

            def forward(self, x):
                x_quantized = torch.quantize_per_tensor(
                    x, 1.0, 0, torch.quint8)
                conv = self.conv(x_quantized)
                return torch.dequantize(conv)

        ref_net = None
        if not relu:
            op_name = "quantized::conv2d"
        else:
            op_name = "quantized::conv2d_relu"

        return quantized_conv2d(), ref_net, op_name

    @pytest.mark.parametrize(
        "params",
        [
            {"weights_shape": [1, 3, 3, 3], "strides": 1,
                "pads": 0, "dilations": 1, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 1,
                "pads": 0, "dilations": 1, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 2,
                "pads": 0, "dilations": 1, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 1,
                "pads": 1, "dilations": 1, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 1,
                "pads": 0, "dilations": 2, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 1,
                "pads": [0, 1], "dilations": 1, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 1,
                "pads": [1, 0], "dilations": 1, "groups": 1},
            {"weights_shape": [3, 1, 3, 3], "strides": 1,
                "pads": 0, "dilations": 1, "groups": 3},
        ],
    )
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("relu", [True, False])
    @pytest.mark.parametrize("scale", [1, 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_quantized_conv2d(self, params, bias, relu, scale, zero_point, ie_device, precision, ir_version):
        self._test(
            *self.create_model(**params, bias=bias, relu=relu,
                               scale=scale, zero_point=zero_point),
            ie_device, precision, ir_version, trace_model=True, freeze_model=False, quantized_ops=True, quant_size=scale
        )
