# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest

np.random.seed(0)
torch.manual_seed(0)


class TestSmoothL1Loss(PytorchLayerTest):
    def _prepare_input(self):
        input_shape = (2, 3, 8, 8)
        return (np.random.uniform(-0.1, 0.1, size=input_shape).astype(np.float32),
                np.random.uniform(-0.1, 0.1, size=input_shape).astype(np.float32))

    def _tolerances(self, precision):
        return (1e-4, 1e-4) if str(precision).upper() == "FP32" else (3e-3, 3e-3)

    def create_model(self, reduction="mean", beta=1.0):
        class aten_smooth_l1_loss(torch.nn.Module):
            def __init__(self, reduction, beta):
                super(aten_smooth_l1_loss, self).__init__()
                self.reduction = reduction
                self.beta = beta

            def forward(self, input, target):
                return F.smooth_l1_loss(input, target, reduction=self.reduction, beta=self.beta)

        ref_net = None
        kind = "aten::smooth_l1_loss"  # keep smooth_l1 kind for all betas
        return aten_smooth_l1_loss(reduction, beta).eval(), ref_net, kind

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_core(self, reduction, ie_device, precision, ir_version):
        self._test(*self.create_model(reduction, 1.0), ie_device, precision, ir_version)

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_beta_zero_matches_l1(self, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__(); self.reduction = reduction
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, beta=0.0, reduction=self.reduction)
        model = M(reduction).eval()
        x = np.random.uniform(-0.1, 0.1, size=(4, 5)).astype(np.float32)
        y = np.random.uniform(-0.1, 0.1, size=(4, 5)).astype(np.float32)
        rtol, atol = self._tolerances(precision)
        def _cmp(ov_outs, pt_outs):
            np.testing.assert_allclose(np.asarray(ov_outs), np.asarray(pt_outs), rtol=rtol, atol=atol)
        self._test(model, None, "aten::l1_loss", ie_device, precision, ir_version,
                   inputs=[x, y], compare_func=_cmp)

    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_smooth_l1_broadcasting(self, reduction, ie_device, precision, ir_version):
        class M(torch.nn.Module):
            def __init__(self, reduction):
                super().__init__(); self.reduction = reduction
            def forward(self, x, y):
                return F.smooth_l1_loss(x, y, reduction=self.reduction, beta=0.5)
        model = M(reduction).eval()
        x = np.random.uniform(-0.1, 0.1, size=(2, 3, 8, 8)).astype(np.float32)
        y = np.random.uniform(-0.1, 0.1, size=(1, 3, 1, 1)).astype(np.float32)
        self._test(model, None, "aten::smooth_l1_loss", ie_device, precision, ir_version, inputs=[x, y])