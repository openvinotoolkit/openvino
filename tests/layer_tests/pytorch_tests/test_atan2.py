# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

@pytest.mark.parametrize("input_shape_rhs", [
    [2, 5, 3, 4],
    [1, 5, 3, 4],
    [1]
])
class TestAtan2(PytorchLayerTest):
    
    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32), self.input_rhs)
    
    def create_model(self):
        
        class aten_atan2(torch.nn.Module):
            def __init__(self):
                super(aten_atan2, self).__init__()
            
            def forward(self, lhs, rhs):
                return torch.arctan2(lhs, rhs)
        
        ref_net = None
        
        return aten_atan2(), ref_net, "aten::atan2"
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_atan2(self, ie_device, precision, ir_version, input_shape_rhs):
        self.input_rhs = np.random.randn(*input_shape_rhs).astype(np.float32)
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)
                