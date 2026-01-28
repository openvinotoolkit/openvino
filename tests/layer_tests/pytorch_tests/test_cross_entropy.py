# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestCrossEntropyLoss(PytorchLayerTest):
    def _prepare_input(self):
        np.random.seed(42)
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,)).astype(np.int64)
        return (logits, targets)

    def create_model(self, reduction="mean", use_weight=False, ignore_index=-100):
        class AtenCrossEntropyLoss(torch.nn.Module):
            def __init__(self, reduction, use_weight, ignore_index):
                super().__init__()
                self.reduction = reduction
                self.weight = torch.rand(10) if use_weight else None
                self.ignore_index = ignore_index

            def forward(self, x, target):
                return F.cross_entropy(x, target, weight=self.weight,
                                     reduction=self.reduction, ignore_index=self.ignore_index)

        return AtenCrossEntropyLoss(reduction, use_weight, ignore_index), None, "aten::cross_entropy_loss"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_cross_entropy_loss(self, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("GPU execution encounters OpenCL runtime errors - issue tracked")
        self._test(*self.create_model(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_cross_entropy_reduction(self, ie_device, precision, ir_version, reduction):
        if ie_device == "GPU":
            pytest.skip("GPU execution encounters OpenCL runtime errors - issue tracked")
        self._test(*self.create_model(reduction=reduction), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cross_entropy_with_weight(self, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("GPU execution encounters OpenCL runtime errors - issue tracked")
        self._test(*self.create_model(use_weight=True), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_cross_entropy_ignore_index(self, ie_device, precision, ir_version):
        if ie_device == "GPU":
            pytest.skip("GPU execution encounters OpenCL runtime errors - issue tracked")
        self._test(*self.create_model(ignore_index=0), ie_device, precision, ir_version)
