# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


def inject_edge_values(tensor):
    flat = tensor.reshape(-1)
    flat[0] = 5.0
    flat[1] = 5.0
    flat[2] = float('nan')
    flat[3] = -10.0
    return tensor


class aten_fractional_max_pool(torch.nn.Module):
    def __init__(self, pool_fn, kernel_size, output_size, return_indices, samples_shape) -> None:
        super().__init__()
        self.pool_fn = pool_fn
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.return_indices = return_indices
        gen = torch.Generator().manual_seed(42)
        self.random_samples = torch.rand(*samples_shape, generator=gen)

    def forward(self, input_tensor):
        return self.pool_fn(input_tensor, self.kernel_size, output_size=self.output_size,
                            return_indices=self.return_indices, _random_samples=self.random_samples)


class TestFractionalMaxPool2D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, kernel_size, output_size, return_indices):
        unbatched = self.input_tensor.ndim == 3
        batch = 1 if unbatched else self.input_tensor.shape[0]
        channels = self.input_tensor.shape[-3]
        model = aten_fractional_max_pool(F.fractional_max_pool2d, kernel_size, output_size, return_indices,
                                         (batch, channels, 2))
        return model, "aten::fractional_max_pool2d"

    @pytest.mark.parametrize('input_shape,kernel_size,output_size', [
        ([2, 1, 4, 4], [2, 2], [2, 2]),
        ([2, 1, 4, 4], [2, 2], [3, 3]),
        ([1, 3, 32, 32], [3, 3], [4, 4]),
        ([1, 3, 32, 32], [2, 3], [16, 10]),
        ([3, 32, 32], [2, 2], [4, 4]),
        ([1, 3, 12, 14], [2, 3], [3, 4]),
        ([1, 3, 12, 14], [3, 2], [1, 1]),
        ([1, 3, 12, 14], [3, 3], [10, 12]),
    ])
    @pytest.mark.parametrize('return_indices', [False, True])
    @pytest.mark.parametrize('edge_values', [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_fractional_max_pool2d(self, ie_device, precision, ir_version, input_shape, kernel_size, output_size,
                                   return_indices, edge_values):
        self.input_tensor = self.random.randn(*input_shape)
        if edge_values:
            inject_edge_values(self.input_tensor)
        self._test(*self.create_model(kernel_size, output_size, return_indices),
                   ie_device, precision, ir_version, trace_model=True)


class TestFractionalMaxPool3D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, kernel_size, output_size, return_indices):
        unbatched = self.input_tensor.ndim == 4
        batch = 1 if unbatched else self.input_tensor.shape[0]
        channels = self.input_tensor.shape[-4]
        model = aten_fractional_max_pool(F.fractional_max_pool3d, kernel_size, output_size, return_indices,
                                         (batch, channels, 3))
        return model, "aten::fractional_max_pool3d"

    @pytest.mark.parametrize('input_shape,kernel_size,output_size', [
        ([2, 1, 4, 4, 4], [2, 2, 2], [2, 2, 2]),
        ([2, 1, 5, 5, 5], [2, 2, 2], [3, 3, 3]),
        ([1, 3, 10, 12, 14], [2, 3, 2], [2, 3, 4]),
        ([1, 3, 10, 12, 14], [3, 3, 3], [4, 4, 4]),
        ([1, 3, 10, 12, 14], [2, 2, 2], [1, 1, 1]),
        ([3, 8, 8, 8], [2, 2, 2], [4, 4, 4]),
    ])
    @pytest.mark.parametrize('return_indices', [False, True])
    @pytest.mark.parametrize('edge_values', [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_fractional_max_pool3d(self, ie_device, precision, ir_version, input_shape, kernel_size, output_size,
                                   return_indices, edge_values):
        self.input_tensor = self.random.randn(*input_shape)
        if edge_values:
            inject_edge_values(self.input_tensor)
        self._test(*self.create_model(kernel_size, output_size, return_indices),
                   ie_device, precision, ir_version, trace_model=True)
