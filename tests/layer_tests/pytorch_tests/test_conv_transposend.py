# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestConvTranspose2D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 10, 10).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias, output_padding):
        import torch
        import torch.nn.functional as F

        class aten_conv_transpose2d(torch.nn.Module):
            def __init__(self):
                super(aten_conv_transpose2d, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(groups)
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups
                self.output_padding = output_padding

            def forward(self, x):
                return F.conv_transpose2d(x, weight=self.weight, bias=self.bias, stride=self.strides, padding=self.pads, output_padding=self.output_padding, dilation=self.dilations, groups=self.groups)

        ref_net = None

        return aten_conv_transpose2d(), ref_net, "aten::conv_transpose2d"

    @pytest.mark.parametrize("params",
                             [{'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [0, 0],
                               'dilations': [2, 2], 'groups': 1, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                                  0, 0], 'dilations': [1, 1], 'groups': 3, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                                  1, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                                  3, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                                  1, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                                  1, 0], 'dilations': [1, 1], 'groups': 3, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                                  1, 0], 'dilations': [2, 2], 'groups': 3, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [2, 1], 'pads': [
                                  1, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'pads': [
                                  0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'pads': [
                                  0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0]},
                              {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'pads': [
                                  1, 1], 'dilations': [2, 2], 'groups': 1, 'output_padding': [1, 1]},
                              ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_conv_transpose2d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1)


class TestConvTranspose1D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 10).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias, output_padding):
        import torch
        import torch.nn.functional as F

        class aten_conv_transpose1d(torch.nn.Module):
            def __init__(self):
                super(aten_conv_transpose1d, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(groups)
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups
                self.output_padding = output_padding

            def forward(self, x):
                return F.conv_transpose1d(
                    x,
                    weight=self.weight,
                    bias=self.bias,
                    stride=self.strides,
                    padding=self.pads,
                    output_padding=self.output_padding,
                    dilation=self.dilations,
                    groups=self.groups
                )

        ref_net = None

        return aten_conv_transpose1d(), ref_net, "aten::conv_transpose1d"

    @pytest.mark.parametrize("params",
                             [{'weights_shape': [3, 1, 1], 'strides': 1, 'pads': 0, 'dilations': 1, 'groups': 1, 'output_padding': 0},
                              {'weights_shape': [3, 1, 1], 'strides': 1, 'pads': 0,
                                  'dilations': 1, 'groups': 3, 'output_padding': 0},
                              {'weights_shape': [3, 1, 1], 'strides': 1, 'pads': 1,
                                  'dilations': 1, 'groups': 1, 'output_padding': 0},
                              {'weights_shape': [3, 1, 1], 'strides': 1, 'pads': 1,
                                  'dilations': 1, 'groups': 3, 'output_padding': 0},
                              {'weights_shape': [3, 1, 1], 'strides': 1, 'pads': 3,
                                  'dilations': 2, 'groups': 1, 'output_padding': 1},
                              {'weights_shape': [3, 1, 1], 'strides': 1, 'pads': 3,
                                  'dilations': 2, 'groups': 3, 'output_padding': 1},
                              ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_conv_transpose1d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1)


class TestConvTranspose3D(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 10, 10, 4).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias, output_padding):
        import torch
        import torch.nn.functional as F

        class aten_conv_transpose3d(torch.nn.Module):
            def __init__(self):
                super(aten_conv_transpose3d, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias = None
                if bias:
                    self.bias = torch.randn(groups)
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups
                self.output_padding = output_padding

            def forward(self, x):
                return F.conv_transpose3d(
                    x,
                    weight=self.weight,
                    bias=self.bias,
                    stride=self.strides,
                    padding=self.pads,
                    output_padding=self.output_padding,
                    dilation=self.dilations,
                    groups=self.groups
                )

        ref_net = None

        return aten_conv_transpose3d(), ref_net, "aten::conv_transpose3d"

    @pytest.mark.parametrize("params",
                             [{'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [0, 0, 0],
                               'dilations': [2, 2, 2], 'groups': 1, 'output_padding': [0, 0, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
                                  0, 0, 0], 'dilations': [1, 1, 1], 'groups': 3, 'output_padding': [0, 0, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
                                  1, 1, 0], 'dilations': [1, 1, 2], 'groups': 1, 'output_padding': [0, 0, 1]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 2], 'pads': [
                                  3, 1, 0], 'dilations': [4, 4, 4], 'groups': 1, 'output_padding': [1, 1, 1]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
                                  1, 0, 1], 'dilations': [1, 2, 1], 'groups': 1, 'output_padding': [0, 1, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
                                  1, 0, 0], 'dilations': [1, 1, 2], 'groups': 3, 'output_padding': [0, 0, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
                                  1, 0, 0], 'dilations': [2, 2, 1], 'groups': 3, 'output_padding': [0, 0, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 1, 2], 'pads': [
                                  1, 0, 0], 'dilations': [3, 4, 2], 'groups': 1, 'output_padding': [2, 0, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 2, 2], 'pads': [
                                  0, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 2, 2], 'pads': [
                                  0, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0]},
                              {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 2, 2], 'pads': [
                                  1, 1, 2], 'dilations': [2, 2, 2], 'groups': 1, 'output_padding': [1, 1, 0]},
                              ])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_conv_transpose3d(self, params, bias, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1)
