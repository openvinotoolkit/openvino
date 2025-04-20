# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestAffineGridGenerator(PytorchLayerTest):
    def _prepare_input(self):
        return (self.theta,)

    def create_model(self, size, align_corners=False):
        class AffineGridGeneratorModel(torch.nn.Module):
            def __init__(self, size, align_corners):
                super().__init__()
                self.align_corners = align_corners
                self.size = size
                
            def forward(self, theta):
                return torch.nn.functional.affine_grid(theta, self.size, align_corners=self.align_corners)
        
        model = AffineGridGeneratorModel(size, align_corners)
        ref_net = None
        
        return model, ref_net, "aten::affine_grid_generator"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
    @pytest.mark.parametrize("height,width", [(1, 1), (8, 8), (2, 3), (6, 8)])
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_affine_grid_generator_2d(self, batch_size, height, width, align_corners, ie_device, precision, ir_version):
        self.theta = np.random.randn(batch_size, 2, 3).astype(np.float32)
        size = [batch_size, 2, height, width]
        self._test(
            *self.create_model(size=size, align_corners=align_corners),
            ie_device,
            precision,
            ir_version,
        )


    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
    @pytest.mark.parametrize("depth,height,width", [(1, 1, 1), (4, 2 , 6), (5, 8, 8), (2, 3, 4), (6, 8, 10)])
    @pytest.mark.parametrize("align_corners", [True, False])
    def test_affine_grid_generator_3d(self, batch_size, depth, height, width, align_corners, ie_device, precision, ir_version):
        self.theta = np.random.randn(batch_size, 3, 4).astype(np.float32)
        size = [batch_size, 2, depth, height, width]
        self._test(
            *self.create_model(size=size, align_corners=align_corners),
            ie_device,
            precision,
            ir_version,
        )
