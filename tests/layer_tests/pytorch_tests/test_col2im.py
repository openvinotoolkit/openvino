# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import math

from pytorch_layer_test_class import PytorchLayerTest


class TestCol2Im(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, output_size, kernel_size, dilation, padding, stride):
        import torch

        class aten_col2im(torch.nn.Module):
            def __init__(self, output_size, kernel_size, dilation, padding, stride):
                super(aten_col2im, self).__init__()
                self.output_size = output_size
                self.kernel_size = kernel_size
                self.dilation = dilation
                self.padding = padding
                self.stride = stride

            def forward(self, x):
                return torch.nn.functional.fold(
                    x,
                    output_size=self.output_size,
                    kernel_size=self.kernel_size,
                    dilation=self.dilation,
                    padding=self.padding,
                    stride=self.stride
                )

        ref_net = None

        return aten_col2im(output_size, kernel_size, dilation, padding, stride), ref_net, "aten::col2im"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("output_size,kernel_size", [([4, 5], [2, 2])])
    @pytest.mark.parametrize("dilation", [1, 2, [1, 2]])
    @pytest.mark.parametrize("padding", [0, 5, [2, 3]])
    @pytest.mark.parametrize("stride", [1, 2, [2, 1]])
    def test_col2im(self, output_size, kernel_size, dilation, padding, stride, ie_device, precision, ir_version):
        d = dilation if isinstance(dilation, list) else [dilation, dilation]
        s = stride if isinstance(stride, list) else [stride, stride]
        p = padding if isinstance(padding, list) else [padding, padding]
        L = 1
        for i in range(2):
            L *= math.floor((output_size[i] + 2 * p[i] - d[i]
                            * (kernel_size[i] - 1) - 1) / float(s[i]) + 1)
        self._test(*self.create_model(output_size, kernel_size,
                   dilation, padding, stride), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": [10, 3 * kernel_size[0] * kernel_size[1], int(L)]})
