# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform

import pytest
import numpy as np
import torch

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from pytorch_layer_test_class import PytorchLayerTest

from torch.ao.nn.quantized import functional as qF

class TestQuantizedConv1D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(2, 3, 25).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch

        class aten_quantized_conv1d(torch.nn.Module):
            def __init__(self):
                super(aten_quantized_conv1d, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return qF.conv1d(x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.groups)

        ref_net = None

        return aten_quantized_conv1d(), ref_net, "quantized::conv1d"

    @pytest.mark.parametrize("params",
                             [{'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 2, 'pads': 0, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 1, 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 0, 'dilations': 2, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 'same', 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 3, 3], 'strides': 1, 'pads': 'valid', 'dilations': 1, 'groups': 1},
                              {'weights_shape': [3, 1, 3], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 3},
                              ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("relu", [True, False])
    @pytest.mark.parametrize("scale", [1, 0.3, 1.3])
    @pytest.mark.parametrize("zero_point", [0, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_conv1d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version)



