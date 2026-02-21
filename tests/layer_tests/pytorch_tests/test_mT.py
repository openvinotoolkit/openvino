# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestMT(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    def create_model(self):
        class aten_mT(torch.nn.Module):
            def forward(self, x):
                return x.mT

        ref_net = None
        return aten_mT(), ref_net, "aten::mT"

    @pytest.mark.parametrize("input_shape", [
        [3, 4],           # 2D matrix
        [2, 3, 4],        # 3D tensor
        [2, 3, 4, 5],     # 4D tensor
        [1, 2, 3, 4, 5],  # 5D tensor
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_mT(self, input_shape, ie_device, precision, ir_version):
        """Test mT which swaps last 2 dimensions"""
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        
        self._test(
            *self.create_model(),
            ie_device, precision, ir_version,
            trace_model=True
        )

    @pytest.mark.parametrize("input_shape", [
        [5],    # 1D - should return unchanged
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_mT_1d(self, input_shape, ie_device, precision, ir_version):
        """Test mT on 1D tensor (should return unchanged)"""
        self.input_tensor = np.random.randn(*input_shape).astype(np.float32)
        
        self._test(
            *self.create_model(),
            ie_device, precision, ir_version,
            trace_model=True
        )
