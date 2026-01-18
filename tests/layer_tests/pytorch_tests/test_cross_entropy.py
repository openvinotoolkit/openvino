# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestCrossEntropyLoss(PytorchLayerTest):
    def _prepare_input(self):
        logits = np.random.randn(4, 10).astype(np.float32)
        targets = np.random.randint(0, 10, size=(4,)).astype(np.int64)
        return (logits, targets)

    def create_model(self):
        class AtenCrossEntropyLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, target):
                # Default ATen behavior:
                # weight=None, reduction='mean', ignore_index=-100, label_smoothing=0.0
                return F.cross_entropy(x, target)

        return AtenCrossEntropyLoss(), None, "aten::cross_entropy_loss"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_cross_entropy_loss(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
        )
