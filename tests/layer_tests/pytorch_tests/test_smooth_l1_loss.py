# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestSmoothL1Loss(PytorchLayerTest):
    def _prepare_input(self):
        # Generate random input data with smaller values to avoid FP16 overflow
        input_shape = (1, 3, 224, 224)
        return (np.random.uniform(-0.1, 0.1, size=input_shape).astype(np.float32),
                np.random.uniform(-0.1, 0.1, size=input_shape).astype(np.float32))

    def create_model(self, reduction="mean", beta=1.0):
        class aten_smooth_l1_loss(torch.nn.Module):
            def __init__(self, reduction, beta):
                super(aten_smooth_l1_loss, self).__init__()
                self.reduction = reduction
                self.beta = beta

            def forward(self, input, target):
                return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)

        ref_net = None
        return aten_smooth_l1_loss(reduction, beta), ref_net, "aten::smooth_l1_loss"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
    def test_smooth_l1_loss(self, reduction, beta, ie_device, precision, ir_version):
        self._test(*self.create_model(reduction, beta), ie_device, precision, ir_version) 