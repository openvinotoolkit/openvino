# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np


class TestCallMethod:
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_callmethod_size(self, ie_device, precision, ir_version):
        """Test prim::CallMethod with tensor.size()"""
        import openvino as ov
        
        class SizeModel(torch.nn.Module):
            def forward(self, x):
                # This will generate prim::CallMethod[name="size"]
                return x.size(0)
        
        # Create test input
        test_input = torch.randn(3, 4, 5)
        
        # Script the model
        model = SizeModel()
        scripted_model = torch.jit.script(model)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(scripted_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        # Note: prim::CallMethod uses framework fallback (translate_pythonop)
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_callmethod_dim(self, ie_device, precision, ir_version):
        """Test prim::CallMethod with tensor.dim()"""
        import openvino as ov
        
        class DimModel(torch.nn.Module):
            def forward(self, x):
                # This will generate prim::CallMethod[name="dim"]
                return x.dim()
        
        # Create test input
        test_input = torch.randn(2, 3, 4)
        
        # Script the model
        model = DimModel()
        scripted_model = torch.jit.script(model)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(scripted_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)
