# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestMaxUnpool2d(PytorchLayerTest):
    def _prepare_input(self):
        return (self.pooled_tensor, self.indices_tensor)

    def create_model(self, kernel_size, stride, padding, output_size):
        class aten_max_unpool2d(torch.nn.Module):
            def __init__(self):
                super(aten_max_unpool2d, self).__init__()
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.output_size = output_size

            def forward(self, x, indices):
                return torch.nn.functional.max_unpool2d(
                    x, indices, self.kernel_size, self.stride, self.padding, self.output_size
                )

        ref_net = None
        return aten_max_unpool2d(), ref_net, "aten::max_unpool2d"

    @pytest.mark.parametrize("input_shape", [[1, 1, 4, 4], [1, 3, 8, 8], [2, 2, 6, 6]])
    @pytest.mark.parametrize("kernel_size", [2, 3])
    @pytest.mark.parametrize("stride", [None, 2])
    @pytest.mark.parametrize("padding", [0, 1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_max_unpool2d(self, input_shape, kernel_size, stride, padding, ie_device, precision, ir_version):
        # First, run max_pool2d to get pooled output and indices
        input_tensor = torch.randn(*input_shape)
        
        # Use stride = kernel_size if stride is None (PyTorch default)
        actual_stride = stride if stride is not None else kernel_size
        
        # Run max_pool2d with return_indices=True
        pooled, indices = torch.nn.functional.max_pool2d(
            input_tensor, kernel_size, actual_stride, padding, return_indices=True
        )
        
        # Calculate output_size for max_unpool2d (original input size)
        output_size = input_tensor.shape[2:]
        
        self.pooled_tensor = pooled.numpy()
        self.indices_tensor = indices.numpy()
        
        self._test(
            *self.create_model(kernel_size, actual_stride, padding, output_size),
            ie_device, precision, ir_version,
            trace_model=True
        )
