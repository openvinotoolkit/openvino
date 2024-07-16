# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestIm2Col(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(10, 3, 24, 24).astype(np.float32),)

    def create_model(self, kernel_size, dilation, padding, stride):
        import torch

        class aten_im2col(torch.nn.Module):
            def __init__(self, kernel_size, dilation, padding, stride):
                super(aten_im2col, self).__init__()
                self.kernel_size = kernel_size
                self.dilation = dilation
                self.padding = padding
                self.stride = stride

            def forward(self, x):
                return torch.nn.functional.unfold(
                    x,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    padding=self.padding,
                    stride=self.stride
                )

        ref_net = None

        return aten_im2col(kernel_size, dilation, padding, stride), ref_net, "aten::im2col"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("kernel_size", [[2, 3], [3, 2], [3, 3], [2, 2], [1, 1]])
    @pytest.mark.parametrize("dilation", [1, 2, 3, (1, 2)])
    @pytest.mark.parametrize("padding", [0, 5, 1, [2, 3]])
    @pytest.mark.parametrize("stride", [3, 1, [2, 1]])
    def test_im2col(self, kernel_size, dilation, padding, stride, ie_device, precision, ir_version):
        self._test(*self.create_model(kernel_size, dilation, padding, stride), ie_device, precision, ir_version)
