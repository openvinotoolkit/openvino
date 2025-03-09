# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestRad2Deg(PytorchLayerTest):
    def _prepare_input(self):
        """
        Generates random test inputs in radians.
        """
        return (np.random.uniform(-np.pi, np.pi, (5, 5)).astype(np.float32),)

    def create_model(self):
        """
        Defines a PyTorch model that applies the radian-to-degree conversion.
        """
        class AtenRad2Deg(torch.nn.Module):
            def forward(self, x):
                return torch.rad2deg(x)

        ref_net = None  # No reference model needed
        return AtenRad2Deg(), ref_net, "aten::rad2deg"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_rad2deg(self, ie_device, precision, ir_version):
        """
        Executes the rad2deg test on different devices and precision levels.
        """
        self._test(*self.create_model(), ie_device, precision, ir_version)
