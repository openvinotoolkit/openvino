# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest

d2_params = [{'weights_shape': [3, 3, 2, 2], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [1, 1], 'groups': 1,
              'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 2, 2], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [
                 1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [
                 1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [0, 0], 'dilations': [
                 1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'bias_shape': [1], 'pads': [
                 1, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [1, 1], 'pads': [
                 1, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'bias_shape': [1], 'pads': [
                 3, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [1, 1], 'pads': [
                 3, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'bias_shape': [1], 'pads': [
                 1, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [1, 1], 'pads': [
                 0, 1], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 1, 0], 'dilations': [1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 0, 1], 'dilations': [1, 1], 'groups': 3, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 1, 0], 'dilations': [2, 2], 'groups': 3, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [1, 1], 'pads': [
                 0, 0], 'dilations': [2, 2], 'groups': 3, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 1], 'bias_shape': [1], 'pads': [
                 1, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [2, 1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'bias_shape': [1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 3, 1, 1], 'strides': [2, 2], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 3, 1, 1], 'strides': [2, 1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': False},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'bias_shape': [1], 'pads': [
                 0, 0], 'dilations': [1, 1], 'groups': 1, 'output_padding': [0, 0], 'transposed': True},
             {'weights_shape': [3, 1, 1, 1], 'strides': [2, 2], 'bias_shape': [1], 'pads': [
                 1, 1], 'dilations': [2, 2], 'groups': 1, 'output_padding': [1, 1], 'transposed': True},
             ]

d1_params = [
    {'weights_shape': [3, 3, 2], 'strides': [1], 'pads': [0], 'dilations': [1], 'groups': 1, 'output_padding': [0],
     'transposed': True},
    {'weights_shape': [3, 3, 2], 'strides': [1], 'pads': [0], 'dilations': [
        1], 'groups': 1, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': [0], 'dilations': [
        1], 'groups': 3, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': [0], 'dilations': [
        1], 'groups': 3, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'bias_shape': [1], 'pads': [
        1], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 3, 1], 'strides': [1], 'pads': [
        1], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'bias_shape': [1], 'pads': [
        3], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 3, 1], 'strides': [1], 'pads': [
        3], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'bias_shape': [1], 'pads': [
        1], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 3, 1], 'strides': [1], 'pads': [
        0], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': [
        1], 'dilations': [1], 'groups': 3, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': [1], 'dilations': [
        1], 'groups': 3, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': [
        1], 'dilations': [2], 'groups': 3, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 1, 1], 'strides': [1], 'pads': [
        0], 'dilations': [2], 'groups': 3, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [2], 'bias_shape': [1], 'pads': [
        1], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 3, 1], 'strides': [2], 'pads': [
        0], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [2], 'bias_shape': [1], 'pads': [
        0], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 3, 1], 'strides': [2], 'pads': [
        0], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 3, 1], 'strides': [1], 'pads': [0], 'dilations': [
        1], 'groups': 1, 'output_padding': [0], 'transposed': False},
    {'weights_shape': [3, 1, 1], 'strides': [2], 'bias_shape': [1], 'pads': [
        0], 'dilations': [1], 'groups': 1, 'output_padding': [0], 'transposed': True},
    {'weights_shape': [3, 1, 1], 'strides': [2], 'bias_shape': [1], 'pads': [
        1], 'dilations': [2], 'groups': 1, 'output_padding': [1], 'transposed': True},
    ]

d3_params = [
    {'weights_shape': [3, 3, 2, 2, 1], 'strides': [1, 1, 1], 'pads': [0, 0, 0], 'dilations': [1, 1, 1], 'groups': 1,
     'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 3, 2, 2, 1], 'strides': [1, 1, 1], 'pads': [0, 0, 0], 'dilations': [
        1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [0, 0, 0], 'dilations': [
        1, 1, 1], 'groups': 3, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [0, 0, 0], 'dilations': [
        1, 1, 1], 'groups': 3, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'bias_shape': [1], 'pads': [
        1, 1, 1], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 3, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
        1, 1, 1], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'bias_shape': [1], 'pads': [
        3, 1, 3], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 3, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
        3, 1, 3], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'bias_shape': [1], 'pads': [
        1, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 3, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
        0, 1, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
        1, 0, 0], 'dilations': [1, 1, 1], 'groups': 3, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
        0, 1, 1], 'dilations': [1, 1, 1], 'groups': 3, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
        1, 0, 0], 'dilations': [2, 2, 1], 'groups': 3, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [1, 1, 1], 'pads': [
        0, 0, 0], 'dilations': [2, 2, 2], 'groups': 3, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 1, 1], 'bias_shape': [1], 'pads': [
        1, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 3, 1, 1, 1], 'strides': [2, 1, 1], 'pads': [
        0, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 2, 2], 'bias_shape': [1], 'pads': [
        0, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 3, 1, 1, 1], 'strides': [2, 2, 2], 'pads': [
        0, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 3, 1, 1, 1], 'strides': [2, 1, 1], 'pads': [
        0, 0, 1], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': False},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 2, 2], 'bias_shape': [1], 'pads': [
        0, 0, 0], 'dilations': [1, 1, 1], 'groups': 1, 'output_padding': [0, 0, 0], 'transposed': True},
    {'weights_shape': [3, 1, 1, 1, 1], 'strides': [2, 2, 2], 'bias_shape': [1], 'pads': [
        1, 1, 1], 'dilations': [2, 2, 2], 'groups': 1, 'output_padding': [1, 1, 1], 'transposed': True},
    ]


class TestConvolution(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        import numpy as np
        shape = (1, 3, 10, 10, 10)
        return (np.random.randn(*shape[:ndim]).astype(np.float32),)

    def create_model(self, weights_shape, strides, pads, dilations, groups, bias, transposed, output_padding=0,
                     bias_shape=None, underscore=True):

        import torch

        bias_dim = 0

        class aten__convolution(torch.nn.Module):
            def __init__(self):
                super(aten__convolution, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias_shape = bias_shape
                if self.bias_shape is None:
                    self.bias_shape = weights_shape[bias_dim]
                self.bias = torch.randn(self.bias_shape) if bias else None
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups
                self.transposed = transposed
                self.output_padding = output_padding
                self._op = torch._convolution

            def forward(self, x):
                return self._op(
                    x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.transposed,
                    self.output_padding, self.groups, False, False, False, False
                )

        class aten_convolution(torch.nn.Module):
            def __init__(self):
                super(aten_convolution, self).__init__()
                self.weight = torch.randn(weights_shape)
                self.bias_shape = bias_shape
                if self.bias_shape is None:
                    self.bias_shape = weights_shape[bias_dim]
                self.bias = torch.randn(self.bias_shape) if bias else None
                self.strides = strides
                self.pads = pads
                self.dilations = dilations
                self.groups = groups
                self.transposed = transposed
                self.output_padding = output_padding
                self._op = torch.convolution

            def forward(self, x):
                return self._op(
                    x, self.weight, self.bias, self.strides, self.pads, self.dilations, self.transposed,
                    self.output_padding, self.groups
                )

        ref_net = None
        if underscore:
            return aten__convolution(), ref_net, "aten::_convolution"
        return aten_convolution(), ref_net, "aten::convolution"

    @pytest.mark.parametrize("params", d1_params)
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("underscore", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    def test_convolution1d(self, params, bias, underscore, ie_device, precision, ir_version):
        if ie_device == "GPU" and params["dilations"] != [1]:
            pytest.xfail(reason="Unsupported dilations of Convolution on GPU")
        self._test(*self.create_model(**params, bias=bias, underscore=underscore),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1,
                   kwargs_to_prepare_input={'ndim': 3})

    @pytest.mark.parametrize("params", d2_params)
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("underscore", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    def test_convolution2d(self, params, bias, underscore, ie_device, precision, ir_version):
        if ie_device == "GPU" and params["dilations"] != [1, 1]:
            pytest.xfail(reason="Unsupported dilations of Convolution on GPU")
        self._test(*self.create_model(**params, bias=bias, underscore=underscore),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1)

    @pytest.mark.parametrize("params", d3_params)
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("underscore", [True, False])
    @pytest.mark.nightly
    def test_convolution3d(self, params, bias, underscore, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias, underscore=underscore),
                   ie_device, precision, ir_version, dynamic_shapes=params['groups'] == 1,
                   kwargs_to_prepare_input={'ndim': 5})
