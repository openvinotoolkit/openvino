# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestLogCumSumExp(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, axis):
        class aten_logcumsumexp(torch.nn.Module):
            def __init__(self, axis):
                super(aten_logcumsumexp, self).__init__()
                self.axis = axis

            def forward(self, x):
                return torch.logcumsumexp(x, self.axis)

        ref_net = None

        return aten_logcumsumexp(axis), ref_net, "aten::logcumsumexp"

    @pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export

    def test_logcumsumexp(self, axis, ie_device, precision, ir_version):
        self._test(*self.create_model(axis), ie_device, precision, ir_version)
