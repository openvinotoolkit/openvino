# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_logit(torch.nn.Module):
    def __init__(self, op_type, eps=None):
        super().__init__()
        self.eps = eps
        self.op_type = op_type

    def forward(self, input_tensor):
        if self.op_type == "aten::logit":
            return torch.logit(input_tensor, eps=self.eps)
        else:
            return torch.special.logit(input_tensor, eps=self.eps)

class TestLogit(PytorchLayerTest):
    def _prepare_input(self):
        # Generate values in (0, 1) for logit function
        return (np.random.uniform(0.1, 0.9, (2, 3, 4)).astype(np.float32),)

    @pytest.mark.parametrize("eps", [None, 1e-6, 0.01])
    @pytest.mark.parametrize("op_type", ["aten::logit", "aten::special_logit"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_logit(self, eps, op_type, ie_device, precision, ir_version):
        self._test(aten_logit(op_type, eps), None, op_type,
                   ie_device, precision, ir_version)
