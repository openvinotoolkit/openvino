# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class UninitializedModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, cond: bool):
        if cond:
            y = x + 1.0
        else:
            y = x - 1.0
        return y

class TestUninitialized(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3).astype("float32"), np.array(True, dtype=bool))

    def create_model(self):
        return UninitializedModel(), None, None

    @pytest.mark.parametrize("input_shape", [(1, 3)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_uninitialized(self, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)