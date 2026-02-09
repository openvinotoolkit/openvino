# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestBinaryCrossEntropyWithLogits(PytorchLayerTest):
    def _prepare_input(self):
        # Deterministic values including extremes that would produce -inf
        # if naive Log(Sigmoid(x)) were used instead of stable -SoftPlus(-x).
        logits = np.array([[-1000.0, 1000.0, -50.0, 50.0],
                           [-1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        target = np.array([[1.0, 0.0, 1.0, 0.0],
                           [0.5, 0.5, 0.0, 1.0]], dtype=np.float32)
        return (logits, target)

    def create_model(self, reduction, with_weight, with_pos_weight):
        import torch
        import torch.nn.functional as F

        class BCEWithLogitsModel(torch.nn.Module):
            def __init__(self, reduction, with_weight, with_pos_weight):
                super().__init__()
                self.reduction = reduction
                self.with_weight = with_weight
                self.with_pos_weight = with_pos_weight

            def forward(self, logits, target):
                weight = torch.tensor([[2.0, 1.0, 0.5, 1.0],
                                       [1.0, 1.5, 2.0, 0.5]]) if self.with_weight else None
                pos_weight = torch.tensor([3.0]) if self.with_pos_weight else None
                return F.binary_cross_entropy_with_logits(
                    logits, target,
                    weight=weight,
                    pos_weight=pos_weight,
                    reduction=self.reduction,
                )

        return BCEWithLogitsModel(reduction, with_weight, with_pos_weight), None, \
            "aten::binary_cross_entropy_with_logits"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("with_weight", [False, True])
    @pytest.mark.parametrize("with_pos_weight", [False, True])
    def test_binary_cross_entropy_with_logits(self, reduction, with_weight, with_pos_weight,
                                              ie_device, precision, ir_version):
        self._test(
            *self.create_model(reduction, with_weight, with_pos_weight),
            ie_device, precision, ir_version,
            use_convert_model=True,
            dynamic_shapes=False,
        )
