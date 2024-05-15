# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestAdaptiveMaxPool3D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, output_size=None, return_indices=False):
        class aten_adaptive_max_pool3d(torch.nn.Module):

            def __init__(self, output_size=None, return_indices=False) -> None:
                super().__init__()
                self.output_size = output_size
                self.return_indices = return_indices

            def forward(self, input_tensor):
                if self.return_indices:
                    output, indices = F.adaptive_max_pool3d(input_tensor, self.output_size, True)
                    return output, indices
                return F.adaptive_max_pool3d(input_tensor, self.output_size, False), input_tensor.to(torch.int64)

        ref_net = None

        return aten_adaptive_max_pool3d(output_size, return_indices), ref_net, "aten::adaptive_max_pool3d"

    @pytest.mark.parametrize('input_shape', [[2, 1, 1, 4, 4],
                                             [4, 1, 3, 32, 32],
                                             [1, 3, 32, 32]])
    @pytest.mark.parametrize('output_size', ([
        [2, 2, 2],
        [4, 4, 4],
    ]))
    @pytest.mark.parametrize('return_indices', ([
        False,
        True,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_adaptive_max_pool3d(self, ie_device, precision, ir_version, input_shape, output_size, return_indices):
        if ie_device == "GPU" and len(input_shape) < 5:
            pytest.xfail(reason="Unsupported shape for adaptive pool on GPU")
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(output_size, return_indices), ie_device, precision, ir_version)


class TestAdaptiveMaxPool2D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, output_size=None, return_indices=False):
        class aten_adaptive_max_pool2d(torch.nn.Module):

            def __init__(self, output_size=None, return_indices=False) -> None:
                super().__init__()
                self.output_size = output_size
                self.return_indices = return_indices

            def forward(self, input_tensor):
                if self.return_indices:
                    output, indices = F.adaptive_max_pool2d(input_tensor, self.output_size, True)
                    return output, indices
                return F.adaptive_max_pool2d(input_tensor, self.output_size, False), input_tensor.to(torch.int64)

        ref_net = None

        return aten_adaptive_max_pool2d(output_size, return_indices), ref_net, "aten::adaptive_max_pool2d"

    @pytest.mark.parametrize('input_shape', [[2, 1, 4, 4],
                                             [1, 3, 32, 32],
                                             [3, 32, 32]])
    @pytest.mark.parametrize('output_size', ([
        [2, 2],
        [4, 4],
    ]))
    @pytest.mark.parametrize('return_indices', ([
        False,
        True,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_adaptive_max_pool2d(self, ie_device, precision, ir_version, input_shape, output_size, return_indices):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(output_size, return_indices), ie_device, precision, ir_version)


class TestAdaptiveMaxPool1D(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self, output_size=None, return_indices=False):
        class aten_adaptive_max_pool1d(torch.nn.Module):

            def __init__(self, output_size=None, return_indices=False) -> None:
                super().__init__()
                self.output_size = output_size
                self.return_indices = return_indices

            def forward(self, input_tensor):
                if self.return_indices:
                    output, indices = F.adaptive_max_pool1d(input_tensor, self.output_size, True)
                    return output, indices
                return F.adaptive_max_pool1d(input_tensor, self.output_size, False), input_tensor.to(torch.int64)

        ref_net = None

        return aten_adaptive_max_pool1d(output_size, return_indices), ref_net, "aten::adaptive_max_pool1d"

    @pytest.mark.parametrize('input_shape', [
        [1, 4, 4],
        [3, 32, 32],
        [16, 8]
    ])
    @pytest.mark.parametrize('output_size', ([
        2,
        4,
    ]))
    @pytest.mark.parametrize('return_indices', ([
        False,
        True,
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_adaptive_max_pool1d(self, ie_device, precision, ir_version, input_shape, output_size, return_indices):
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        self._test(*self.create_model(output_size, return_indices), ie_device, precision, ir_version)
