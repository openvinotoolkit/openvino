# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('alpha', (-0.5, 0, 0.5, 1, 2))
@pytest.mark.parametrize('input_rhs', (np.random.randn(2, 5, 3, 4).astype(np.float32),
                                       np.random.randn(1, 5, 3, 4).astype(np.float32),
                                       np.random.randn(1).astype(np.float32)))
class TestAdd(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32), self.input_rhs)

    def create_model(self, alpha):
        class aten_add(torch.nn.Module):

            def __init__(self, alpha) -> None:
                super().__init__()
                self.alpha = alpha

            def forward(self, lhs, rhs):
                return torch.add(lhs, rhs, alpha=self.alpha)

        ref_net = None

        return aten_add(alpha), ref_net, "aten::add"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_add(self, ie_device, precision, ir_version, alpha, input_rhs):
        self.input_rhs = input_rhs
        self._test(*self.create_model(alpha), ie_device, precision, ir_version)
