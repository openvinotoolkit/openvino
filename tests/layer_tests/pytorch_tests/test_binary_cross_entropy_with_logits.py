# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestBinaryCrossEntropyWithLogits(PytorchLayerTest):
    def _prepare_input(self):
        input_tensor = np.random.randn(*self.input_shape).astype(np.float32)
        target_tensor = np.random.uniform(0, 1, self.input_shape).astype(np.float32)
        return (input_tensor, target_tensor)

    def create_model(self, input_shape):
        self.input_shape = input_shape

        class BCEWithLogitsModule(torch.nn.Module):
            def forward(self, input, target):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target)

        ref_net = None
        op_name = "aten::binary_cross_entropy_with_logits"
        return BCEWithLogitsModule(), ref_net, op_name

    @pytest.mark.parametrize("input_shape", [
        [2, 3],
        [1, 3, 224, 224],
        [4, 5, 6],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_bce_with_logits(self, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(input_shape),
                   ie_device, precision, ir_version)
