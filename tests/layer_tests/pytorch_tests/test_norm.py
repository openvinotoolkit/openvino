# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import torch


@pytest.mark.parametrize('p', [-2, -1, 0, 1, 2, 2.5, float('inf'), float('-inf')])
@pytest.mark.parametrize('dim', [[0], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize('keepdim', [True, False])
class TestNorm(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 3, 4, 5), )

    def create_model(self, p, dim, keepdim):
        class aten_norm(torch.nn.Module):

            def __init__(self, p, dim, keepdim) -> None:
                super().__init__()
                self.p = p
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, input_data):
                return torch._VF.norm(input_data, self.p, self.dim, self.keepdim)

        ref_net = None

        return aten_norm(p, dim, keepdim), ref_net, "aten::norm"

    @pytest.mark.nightly
    def test_norm(self, ie_device, precision, ir_version, p, dim, keepdim):
        self._test(*self.create_model(p, dim, keepdim),
                   ie_device, precision, ir_version)
