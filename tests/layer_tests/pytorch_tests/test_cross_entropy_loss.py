# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class aten_cross_entropy_loss(torch.nn.Module):
    def __init__(self, num_classes, reduction="mean", weight=None):
        super().__init__()
        self.reduction = reduction
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None
        self.num_classes = num_classes

    def forward(self, input_tensor, target):
        return F.cross_entropy(input_tensor, target, weight=self.weight, reduction=self.reduction)


class TestCrossEntropyLoss(PytorchLayerTest):
    def _prepare_input(self):
        logits = np.random.randn(*self.input_shape).astype(np.float32)
        target_shape = list(self.input_shape)
        target_shape.pop(1)
        target = np.random.randint(0, self.num_classes, size=target_shape).astype(np.int64)
        return (logits, target)

    def _create_model(self, reduction="mean", use_weight=False):
        weight = None
        if use_weight:
            weight = torch.linspace(0.5, 1.5, steps=self.num_classes, dtype=torch.float32)
        model = aten_cross_entropy_loss(self.num_classes, reduction=reduction, weight=weight)
        return model, None, "aten::cross_entropy_loss"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_cross_entropy_loss_basic(self, reduction, ie_device, precision, ir_version):
        self.input_shape = (4, 5)
        self.num_classes = self.input_shape[1]
        self._test(*self._create_model(reduction=reduction), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_cross_entropy_loss_weighted_spatial(self, ie_device, precision, ir_version):
        self.input_shape = (2, 6, 4, 4)
        self.num_classes = self.input_shape[1]
        self._test(*self._create_model(reduction="mean", use_weight=True), ie_device, precision, ir_version)
