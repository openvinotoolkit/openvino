# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestStr(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(2, 3).astype(np.float32),)

    def create_model(self, const_val):
        class aten_str(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def forward(self, x):
                s = str(self.val)
                return torch.tensor(len(s), dtype=torch.float32) + x.sum() * 0

        return aten_str(const_val), None

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("const_val", [0, 42, -7, True, False, 3.14])
    def test_str(self, const_val, ie_device, precision, ir_version):
        self._test(*self.create_model(const_val), ie_device, precision, ir_version)
