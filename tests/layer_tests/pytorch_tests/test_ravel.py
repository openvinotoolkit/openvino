# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np


class TestRavel:
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ravel_basic(self, ie_device, precision, ir_version):
        """Test aten::ravel with 2D tensor"""
        import openvino as ov
        
        class RavelModel(torch.nn.Module):
            def forward(self, x):
                return torch.ravel(x)
        
        # Create test input
        test_input = torch.randn(3, 4)
        
        # Trace the model
        model = RavelModel()
        traced_model = torch.jit.trace(model, test_input)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(traced_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        # Note: aten::ravel uses translate_flatten (flattens to 1D)
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ravel_multidim(self, ie_device, precision, ir_version):
        """Test aten::ravel with multi-dimensional tensor"""
        import openvino as ov
        
        class RavelModel(torch.nn.Module):
            def forward(self, x):
                return torch.ravel(x)
        
        # Create test input
        test_input = torch.randn(2, 3, 4, 5)
        
        # Trace the model
        model = RavelModel()
        traced_model = torch.jit.trace(model, test_input)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(traced_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)
