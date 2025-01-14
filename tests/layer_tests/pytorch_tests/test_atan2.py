# Copyright (C) 2018-2025 Intel Corporation
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
        
class TestAtan2Types(PytorchLayerTest):
    
    def _prepare_input(self):
        return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randn(self.rhs_shape).to(self.rhs_type).numpy())
    
    def create_model(self, lhs_type, rhs_type):
        
        class aten_atan2(torch.nn.Module):
            def __init__(self, lhs_type, rhs_type):
                super(aten_atan2, self).__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type
            
            def forward(self, lhs, rhs):
                return torch.arctan2(lhs.to(self.lhs_type), rhs.to(self.rhs_type))
        
        ref_net = None
        
        return aten_atan2(lhs_type, rhs_type), ref_net, "aten::atan2"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.int, torch.float32],
                              [torch.int, torch.float64],
                              [torch.float32, torch.float64],
                              [torch.int64, torch.float32]
                              ])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], [1, 3]),
                                                          ([3, 2, 3], [2, 3]),
                                                          ])    
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_atan2_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, rhs_type),
                   ie_device, precision, ir_version, freeze_model=False, trace_model=True)    