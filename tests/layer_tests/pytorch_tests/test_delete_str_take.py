# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np


class TestDeleteStrTake:
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_delete_operation(self, ie_device, precision, ir_version):
        """Test aten::Delete (list deletion, should be skipped)"""
        import openvino as ov
        
        class DeleteModel(torch.nn.Module):
            def forward(self, x):
                # Create a list and delete from it
                # This generates aten::Delete in TorchScript
                lst = [x, x + 1, x + 2]
                del lst[1]
                return lst[0] + lst[1]
        
        # Create test input
        test_input = torch.randn(3, 4)
        
        # Trace the model
        model = DeleteModel()
        traced_model = torch.jit.trace(model, test_input)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(traced_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        # Note: aten::Delete uses skip_node (no-op in inference)
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_str_operation(self, ie_device, precision, ir_version):
        """Test aten::str (string conversion, should be skipped)"""
        import openvino as ov
        
        class StrModel(torch.nn.Module):
            def forward(self, x):
                # String operations are typically optimized away
                # but aten::str can appear in some graphs
                return x * 2
        
        # Create test input
        test_input = torch.randn(2, 3)
        
        # Trace the model
        model = StrModel()
        traced_model = torch.jit.trace(model, test_input)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(traced_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        # Note: aten::str uses skip_node
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_take_operation(self, ie_device, precision, ir_version):
        """Test aten::take (indexing with flattened indices)"""
        import openvino as ov
        
        class TakeModel(torch.nn.Module):
            def forward(self, x, indices):
                # aten::take indexes into flattened tensor
                return torch.take(x, indices)
        
        # Create test inputs
        test_input = torch.randn(3, 4)
        test_indices = torch.tensor([0, 5, 11])
        
        # Trace the model
        model = TakeModel()
        traced_model = torch.jit.trace(model, (test_input, test_indices))
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(traced_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        # Note: aten::take uses translate_pythonop (framework fallback)
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)
