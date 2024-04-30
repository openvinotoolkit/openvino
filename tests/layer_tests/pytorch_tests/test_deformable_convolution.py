# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest
from torchvision.ops import deform_conv2d


params = [
    {
        "weights_shape": [64, 64, 3, 3],
        "offset_shape": [1, 18, 64, 64],
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 64, 3, 3],
        "offset_shape": [1, 18, 62, 64],
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (2, 1),
    },
    {
        "weights_shape": [64, 64, 3, 3],
        "offset_shape": [1, 18, 66, 64],
        "stride": (1, 1),
        "padding": (2, 1),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 64, 3, 3],
        "offset_shape": [1, 18, 32, 64],
        "stride": (2, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 64, 3, 3],
        "offset_shape": [1, 18, 62, 62],
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 64, 3, 3],
        "offset_shape": [1, 18, 66, 66],
        "stride": (1, 1),
        "padding": (2, 2),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 16, 3, 3],
        "offset_shape": [1, 18, 64, 64],
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [60, 16, 3, 3],
        "offset_shape": [1, 18, 64, 64],
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 1, 3, 3],
        "offset_shape": [1, 18, 64, 64],
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 64, 3, 3],
        "offset_shape": [1, 36, 64, 64],
        "stride": (1, 1),
        "padding": (1, 1),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [64, 32, 3, 3],
        "offset_shape": [1, 36, 68, 68],
        "stride": (1, 1),
        "padding": (3, 3),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [62, 32, 3, 3],
        "offset_shape": [1, 36, 68, 68],
        "stride": (1, 1),
        "padding": (3, 3),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [2, 32, 3, 3],
        "offset_shape": [1, 36, 68, 68],
        "stride": (1, 1),
        "padding": (3, 3),
        "dilation": (1, 1),
    },
    {
        "weights_shape": [1, 64, 3, 3],
        "offset_shape": [1, 18, 68, 68],
        "stride": (1, 1),
        "padding": (3, 3),
        "dilation": (1, 1),
    },
]


class TestDeformableConvolution(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.rand(1, 64, 64, 64).astype(np.float32),)

    def create_model(
        self,
        offset_shape,
        weights_shape,
        stride,
        padding,
        dilation,
        bias,
        mask,
        mask_shape=None,
        bias_shape=None,
    ):
        class aten_deform_convolution(torch.nn.Module):
            def __init__(self):
                super(aten_deform_convolution, self).__init__()
                self.weight = torch.rand(weights_shape)
                self.offset = torch.rand(offset_shape)
                if mask_shape is None:
                    self.mask_shape = deepcopy(offset_shape)
                    self.mask_shape[1] = self.mask_shape[1] // 2
                else:
                    self.mask_shape = mask_shape
                if mask:
                    self.mask = torch.rand(self.mask_shape)
                else:
                    self.mask = None
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.bias_shape = bias_shape
                if self.bias_shape is None:
                    self.bias_shape = weights_shape[0]
                self.bias = torch.rand(self.bias_shape) if bias else None

            def forward(self, x):
                return deform_conv2d(
                    x,
                    self.offset,
                    self.weight,
                    bias=self.bias,
                    mask=self.mask,
                    stride=self.stride,
                    dilation=self.dilation,
                    padding=self.padding,
                )

        ref_net = None
        return aten_deform_convolution(), ref_net, "torchvision::deform_conv2d"

    @pytest.mark.parametrize("params", params)
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("mask", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_deformable_convolution2d(self, params, bias, mask, ie_device, precision, ir_version):
        self._test(*self.create_model(**params, bias=bias, mask=mask),
                   ie_device, precision, ir_version, trace_model=True,
                   dynamic_shapes=ie_device != "GPU"
                   )
