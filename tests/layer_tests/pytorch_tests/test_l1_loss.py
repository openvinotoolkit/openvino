# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestL1Loss(PytorchLayerTest):
    def _prepare_input(self):
        x = np.random.uniform(-0.1, 0.1, size=(6, 7)).astype(np.float32)
        y = np.random.uniform(-0.1, 0.1, size=(6, 7)).astype(np.float32)
        return (x, y)

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_l1_loss_basic(self, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__()
                self.reduction = reduction
            def forward(self, x, y):
                return F.l1_loss(x, y, reduction=self.reduction)
        model = M(reduction).eval()
        x = np.random.randn(6, 7).astype(np.float32)
        y = np.random.randn(6, 7).astype(np.float32)
        self._test(model, None, "aten::l1_loss", ie_device, precision, ir_version, inputs=[x, y])


class TestL1LossFromSmoothL1(PytorchLayerTest):
    @pytest.mark.nightly
    def test_beta_zero_from_smooth_l1_lowers_to_l1(self, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def forward(self, x, y):
                # TS lowers smooth_l1(beta=0) to aten::l1_loss
                return F.smooth_l1_loss(x, y, beta=0.0, reduction="mean")

        model = M().eval()
        x = np.random.randn(2, 3, 8, 8).astype(np.float32)
        y = np.random.randn(2, 3, 8, 8).astype(np.float32)

        kind = "aten::l1_loss"  
        self._test(model, None, kind, ie_device, precision, ir_version, inputs=[x, y])