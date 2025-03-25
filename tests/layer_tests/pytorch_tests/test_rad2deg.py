# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestRad2Deg(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2)):
        return (np.random.uniform(-np.pi, np.pi, size=input_shape).astype(np.float32),)

    def create_model(self):
        class aten_rad2deg(torch.nn.Module):
            def forward(self, x):
                return x * (180.0 / np.pi)
        ref_net = None  
        return aten_rad2deg(), ref_net, "aten::mul"

    @pytest.mark.parametrize("input_shape", [(2, 2), (3, 3), (4, 5)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_rad2deg(self, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape})