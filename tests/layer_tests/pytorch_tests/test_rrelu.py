# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class aten_rrelu(torch.nn.Module):
    def __init__(self, lower, upper, inplace):
        super(aten_rrelu, self).__init__()
        self.lower = lower 
        self.upper = upper 
        self.inplace = inplace 

    def forward(self, x):
        rrelu = F.rrelu(x, lower=self.lower, upper=self.upper, inplace=self.inplace)

        return x, rrelu


class TestLeakyRelu(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    @pytest.mark.parametrize("lower", [-1e10, -1e-03, -1])
    @pytest.mark.parametrize("upper", [0, 1e-3, 1, 1e10])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_rrelu(self, lower, upper, inplace, ie_device, precision, ir_version):
        model = aten_rrelu(lower, upper, inplace)
        self._test(model, ie_device, precision, ir_version)


