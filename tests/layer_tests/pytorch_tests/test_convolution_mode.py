# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestConv2D(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        import numpy as np
        input_shape = (1, 3, 10, 10, 10)
        return (np.random.randn(*input_shape[:ndim]).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias):
        import torch

        class aten_convolution_mode(torch.nn.Module):
            def __init__(self):
                super(aten_convolution_mode, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(weights_shape[0])
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups

            def forward(self, x):
                return torch._convolution_mode(x, self.weight, self.bias, self.strides, self.pads, self.dilations,
                                               self.groups)

        ref_net = None

        return aten_convolution_mode(), ref_net, "aten::_convolution_mode"

    @pytest.mark.parametrize("params",
                             [
                                 {'weights_shape': [1, 3, 3], 'strides': [1], 'pads': 'same', 'dilations': [1],
                                  'groups': 1},
                                 {'weights_shape': [1, 3, 3], 'strides': [1], 'pads': 'valid', 'dilations': [1],
                                  'groups': 1},
                                 {'weights_shape': [1, 3, 3], 'strides': [1], 'pads': 'same', 'dilations': [2],
                                  'groups': 1},
                                 {'weights_shape': [1, 3, 3], 'strides': [1], 'pads': 'valid', 'dilations': [2],
                                  'groups': 1},
                                 {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': 'same', 'dilations': [1],
                                  'groups': 3},
                                 {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': 'valid', 'dilations': [1],
                                  'groups': 3},
                                 {'weights_shape': [1, 3, 3], 'strides': [2], 'pads': 'valid', 'dilations': [1],
                                  'groups': 1},
                                 {'weights_shape': [1, 3, 3], 'strides': [2], 'pads': 'valid', 'dilations': [2],
                                  'groups': 1},
                                 {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': 'same', 'dilations': [2],
                                  'groups': 3},
                                 {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': 'valid', 'dilations': [2],
                                  'groups': 3},
                             ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_convolution_mode_1d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1,
                   kwargs_to_prepare_input={'ndim': 3})

    @pytest.mark.parametrize("params",
                             [
                                 {'weights_shape': [1, 3, 3, 3], 'strides': [1, 1], 'pads': 'same', 'dilations': [1, 1],
                                  'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3], 'strides': [1, 1], 'pads': 'valid',
                                  'dilations': [1, 1], 'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3], 'strides': [1, 1], 'pads': 'same', 'dilations': [2, 2],
                                  'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3], 'strides': [1, 1], 'pads': 'valid',
                                  'dilations': [2, 2], 'groups': 1},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': 'same', 'dilations': [1, 1],
                                  'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': 'valid',
                                  'dilations': [1, 1], 'groups': 3},
                                 {'weights_shape': [1, 3, 3, 3], 'strides': [2, 2], 'pads': 'valid',
                                  'dilations': [1, 1], 'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3], 'strides': [2, 2], 'pads': 'valid',
                                  'dilations': [2, 2], 'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3], 'strides': [2, 1], 'pads': 'valid',
                                  'dilations': [1, 1], 'groups': 1},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'pads': 'valid',
                                  'dilations': [1, 1], 'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'pads': 'valid',
                                  'dilations': [2, 2], 'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [2, 1], 'pads': 'valid',
                                  'dilations': [1, 1], 'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': 'same', 'dilations': [2, 1],
                                  'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': 'valid',
                                  'dilations': [2, 1], 'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': 'same', 'dilations': [2, 2],
                                  'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': 'valid',
                                  'dilations': [2, 2], 'groups': 3},
                             ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_convolution_mode_2d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1)

    @pytest.mark.parametrize("params",
                             [
                                 {'weights_shape': [1, 3, 3, 3, 3], 'strides': [1, 1, 1], 'pads': 'same',
                                  'dilations': [1, 1, 1], 'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3, 3], 'strides': [1, 1, 1], 'pads': 'valid',
                                  'dilations': [1, 1, 1], 'groups': 1},
                                 {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': 'same',
                                  'dilations': [1, 1, 1], 'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': 'valid',
                                  'dilations': [1, 1, 1], 'groups': 3},
                                 {'weights_shape': [1, 3, 3, 3, 3], 'strides': [2, 2, 1], 'pads': 'valid',
                                  'dilations': [1, 1, 1], 'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3, 3], 'strides': [2, 2, 2], 'pads': 'valid',
                                  'dilations': [1, 1, 1], 'groups': 1},
                                 {'weights_shape': [1, 3, 3, 3, 3], 'strides': [2, 2, 2], 'pads': 'valid',
                                  'dilations': [2, 2, 2], 'groups': 1},
                                 {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': 'same',
                                  'dilations': [2, 1, 2], 'groups': 3},
                                 {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': 'valid',
                                  'dilations': [2, 1, 2], 'groups': 3},
                             ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_convolution_mode_3d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1,
                   kwargs_to_prepare_input={'ndim': 5})
