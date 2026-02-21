# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestAffineGridGenerator(PytorchLayerTest):
    def _prepare_input(self):
        return (self.theta_tensor,)

    def create_model(self, size, align_corners):
        class aten_affine_grid_generator(torch.nn.Module):
            def __init__(self):
                super(aten_affine_grid_generator, self).__init__()
                self.size = size
                self.align_corners = align_corners

            def forward(self, theta):
                return torch.nn.functional.affine_grid(
                    theta, self.size, self.align_corners
                )

        ref_net = None
        return aten_affine_grid_generator(), ref_net, "aten::affine_grid_generator"

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("height,width", [(4, 4), (8, 6), (16, 16)])
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_affine_grid_generator_identity(self, batch_size, height, width, align_corners, ie_device, precision, ir_version):
        """Test with identity transformation (should return normalized grid)"""
        size = (batch_size, 3, height, width)  # N, C, H, W
        
        # Identity affine matrix: [[1, 0, 0], [0, 1, 0]]
        theta = torch.zeros(batch_size, 2, 3)
        theta[:, 0, 0] = 1.0  # scale x
        theta[:, 1, 1] = 1.0  # scale y
        
        self.theta_tensor = theta.numpy()
        
        self._test(
            *self.create_model(size, align_corners),
            ie_device, precision, ir_version,
            trace_model=True
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("height,width", [(8, 8)])
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_affine_grid_generator_translation(self, batch_size, height, width, align_corners, ie_device, precision, ir_version):
        """Test with translation transformation"""
        size = (batch_size, 3, height, width)
        
        # Translation: [[1, 0, 0.5], [0, 1, 0.5]]
        theta = torch.zeros(batch_size, 2, 3)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = 0.5  # translate x
        theta[:, 1, 2] = 0.5  # translate y
        
        self.theta_tensor = theta.numpy()
        
        self._test(
            *self.create_model(size, align_corners),
            ie_device, precision, ir_version,
            trace_model=True
        )

    @pytest.mark.parametrize("batch_size", [1])
    @pytest.mark.parametrize("height,width", [(8, 8)])
    @pytest.mark.parametrize("align_corners", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_affine_grid_generator_scale(self, batch_size, height, width, align_corners, ie_device, precision, ir_version):
        """Test with scaling transformation"""
        size = (batch_size, 3, height, width)
        
        # Scale: [[2, 0, 0], [0, 2, 0]]
        theta = torch.zeros(batch_size, 2, 3)
        theta[:, 0, 0] = 2.0  # scale x
        theta[:, 1, 1] = 2.0  # scale y
        
        self.theta_tensor = theta.numpy()
        
        self._test(
            *self.create_model(size, align_corners),
            ie_device, precision, ir_version,
            trace_model=True
        )
