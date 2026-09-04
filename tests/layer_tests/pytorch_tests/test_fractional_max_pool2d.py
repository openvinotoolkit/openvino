# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest

class TestFractionalMaxPool3D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, kernel_size, output_size=None, return_indices=False):
        class aten_fractional_max_pool3d(torch.nn.Module):

            def __init__(self, kernel_size, output_size=None, return_indices=False, batch_size=1, channels=1) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.output_size = output_size
                self.return_indices = return_indices
                # Generate deterministic random samples for precise equivalence testing
                gen = torch.Generator().manual_seed(42)
                self.random_samples = torch.rand(batch_size, channels, 3, generator=gen)

            def forward(self, input_tensor):
                if self.return_indices:
                    output, indices = F.fractional_max_pool3d(input_tensor, self.kernel_size, self.output_size, return_indices=True, _random_samples=self.random_samples)
                    return output, indices
                return F.fractional_max_pool3d(input_tensor, self.kernel_size, self.output_size, return_indices=False, _random_samples=self.random_samples), input_tensor.to(torch.int64)

        batch_size = 1 if len(self.input_tensor.shape) == 4 else self.input_tensor.shape[0]
        channels = self.input_tensor.shape[0] if len(self.input_tensor.shape) == 4 else self.input_tensor.shape[1]
        return aten_fractional_max_pool3d(kernel_size, output_size, return_indices, batch_size, channels), "aten::fractional_max_pool3d"

    @pytest.mark.parametrize('input_shape', [[2, 1, 4, 4, 4],
                                             [1, 3, 32, 32, 32],
                                             [3, 32, 32, 32],
                                             [1, 3, 10, 12, 14]])
    @pytest.mark.parametrize('kernel_size', ([
        [2, 2, 2],
        [3, 3, 3],
        [2, 3, 2],
    ]))
    @pytest.mark.parametrize('output_size', ([
        [2, 2, 2],
        [4, 4, 4],
        [2, 3, 4],
    ]))
    @pytest.mark.parametrize('return_indices', ([
        False,
        True,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_fractional_max_pool3d(self, ie_device, precision, ir_version, input_shape, kernel_size, output_size, return_indices):
        if ie_device == "GPU" and len(input_shape) < 5:
            pytest.xfail(reason="Unsupported shape for adaptive pool on GPU")
        self.input_tensor = self.random.randn(*input_shape)
        # Inject tie, NaN, and negative values to verify TopK edge cases
        if self.input_tensor.size >= 4:
            flat = self.input_tensor.reshape(-1)
            flat[0] = 5.0
            flat[1] = 5.0  # duplicate maximum
            flat[2] = float('nan')  # NaN propagation
            flat[3] = -10.0  # negative values
        self._test(*self.create_model(kernel_size, output_size, return_indices), ie_device, precision, ir_version, trace_model=True)


class TestFractionalMaxPool2D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, kernel_size, output_size=None, return_indices=False):
        class aten_fractional_max_pool2d(torch.nn.Module):

            def __init__(self, kernel_size, output_size=None, return_indices=False, batch_size=1, channels=1) -> None:
                super().__init__()
                self.kernel_size = kernel_size
                self.output_size = output_size
                self.return_indices = return_indices
                # Generate deterministic random samples for precise equivalence testing
                gen = torch.Generator().manual_seed(42)
                self.random_samples = torch.rand(batch_size, channels, 2, generator=gen)

            def forward(self, input_tensor):
                if self.return_indices:
                    output, indices = F.fractional_max_pool2d(input_tensor, self.kernel_size, self.output_size, return_indices=True, _random_samples=self.random_samples)
                    return output, indices
                return F.fractional_max_pool2d(input_tensor, self.kernel_size, self.output_size, return_indices=False, _random_samples=self.random_samples), input_tensor.to(torch.int64)

        batch_size = 1 if len(self.input_tensor.shape) == 3 else self.input_tensor.shape[0]
        channels = self.input_tensor.shape[0] if len(self.input_tensor.shape) == 3 else self.input_tensor.shape[1]
        return aten_fractional_max_pool2d(kernel_size, output_size, return_indices, batch_size, channels), "aten::fractional_max_pool2d"

    @pytest.mark.parametrize('input_shape', [[2, 1, 4, 4],
                                             [1, 3, 32, 32],
                                             [3, 32, 32],
                                             [1, 3, 12, 14]])
    @pytest.mark.parametrize('kernel_size', ([
        [2, 2],
        [3, 3],
        [2, 3],
    ]))
    @pytest.mark.parametrize('output_size', ([
        [2, 2],
        [4, 4],
        [3, 4],
    ]))
    @pytest.mark.parametrize('return_indices', ([
        False,
        True,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_fractional_max_pool2d(self, ie_device, precision, ir_version, input_shape, kernel_size, output_size, return_indices):
        self.input_tensor = self.random.randn(*input_shape)
        # Inject tie, NaN, and negative values to verify TopK edge cases
        if self.input_tensor.size >= 4:
            flat = self.input_tensor.reshape(-1)
            flat[0] = 5.0
            flat[1] = 5.0  # duplicate maximum
            flat[2] = float('nan')  # NaN propagation
            flat[3] = -10.0  # negative values
        self._test(*self.create_model(kernel_size, output_size, return_indices), ie_device, precision, ir_version, trace_model=True)
