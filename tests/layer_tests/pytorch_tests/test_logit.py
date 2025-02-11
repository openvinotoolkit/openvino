# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class aten_logit(torch.nn.Module):
    def __init__(self, eps=None):
        self.eps = eps

    def forward(self, input_tensor):
        return torch.logit(input_tensor, eps=self.eps)
    
class TestLogit(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 5, 9, 7),)
    
    @pytest.mark.parametrize("eps", [1e-6, 1e-7])

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    
    def test_logit(self, eps, ie_device, precision, ir_version):
        self._test(aten_logit(eps), None, "aten::logit", 
                              ie_device, precision, ir_version)