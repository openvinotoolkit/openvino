# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('input_rhs', (np.random.randn(2, 5, 3, 4).astype(np.float32),
                                       np.random.randn(1, 5, 3, 4).astype(np.float32),
                                       np.random.randn(1).astype(np.float32)))
class TestRemainder(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32), self.input_rhs)

    def create_model(self):
        import torch
        class aten_remainder(torch.nn.Module):

            def forward(self, lhs, rhs):
                return torch.remainder(lhs, rhs)

        ref_net = None

        return aten_remainder(), ref_net, "aten::remainder"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_remainder(self, ie_device, precision, ir_version, input_rhs):
        self.input_rhs = input_rhs
        self._test(*self.create_model(), ie_device, precision, ir_version)