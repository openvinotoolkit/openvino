# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestQuantizedConv2DPrepack(PytorchLayerTest):
    rng = np.random.default_rng(seed=456)

    def _prepare_input(self):
        return (np.round(self.rng.random([2, 3, 25, 25], dtype=np.float32), 4),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        class quantized_conv2d_prepack_test(torch.nn.Module):
            def __init__(self):
                super(quantized_conv2d_prepack_test, self).__init__()
                # create weight and bias
                self.weight = torch.randn(weights_shape)
                if bias:
                    self.bias = torch.randn(weights_shape[0])
                else:
                    self.bias = None
                self.strides = strides if isinstance(strides, (list, tuple)) else [strides, strides]
                self.pads = pads if isinstance(pads, (list, tuple)) else [pads, pads]
                self.dilations = dilations if isinstance(dilations, (list, tuple)) else [dilations, dilations]
                self.groups = groups

            def forward(self, x):
                # quantize input
                x_quantized = torch.quantize_per_tensor(x, 1.0, 0, torch.quint8)
                
                # use conv2d_prepack to pack parameters
                packed_params = torch.ops.quantized.conv2d_prepack(
                    self.weight,
                    self.bias,
                    self.strides,
                    self.pads,
                    self.dilations,
                    self.groups
                )
                
                # use packed params with conv2d
                output = torch.ops.quantized.conv2d(
                    x_quantized,
                    packed_params,
                    1.0,  # output scale
                    0     # output zero point
                )
                
                return torch.dequantize(output)

        ref_net = None
        op_name = "quantized::conv2d_prepack"
        
        return quantized_conv2d_prepack_test(), ref_net, op_name

    @pytest.mark.parametrize(
        "params",
        [
            {"weights_shape": [2, 3, 3, 3], "strides": 1, "pads": 0, "dilations": 1, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 2, "pads": 1, "dilations": 1, "groups": 1},
            {"weights_shape": [2, 3, 3, 3], "strides": 1, "pads": 0, "dilations": 2, "groups": 1},
            {"weights_shape": [3, 1, 3, 3], "strides": 1, "pads": 0, "dilations": 1, "groups": 3},
        ],
    )
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_conv2d_prepack(self, params, bias, ie_device, precision, ir_version):
        self._test(
            *self.create_model(**params, bias=bias),
            ie_device, precision, ir_version, trace_model=True, freeze_model=False, quantized_ops=True
        )
