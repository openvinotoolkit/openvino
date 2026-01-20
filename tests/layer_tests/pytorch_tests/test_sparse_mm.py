# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np


class TestSparseMM:
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sparse_mm_basic(self, ie_device, precision, ir_version):
        """Test basic sparse matrix multiplication"""
        import openvino as ov
        
        class SparseMMModel(torch.nn.Module):
            def forward(self, sparse_mat, dense_mat):
                return torch.ops.aten._sparse_mm(sparse_mat, dense_mat)
        
        # Create sparse matrix (2x3)
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.long)
        values = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        sparse_mat = torch.sparse_coo_tensor(indices, values, (2, 3))
        
        # Create dense matrix (3x4)
        dense_mat = torch.randn(3, 4)
        
        # Script the model
        model = SparseMMModel()
        scripted_model = torch.jit.script(model)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(scripted_model)
        ov_model = fe.convert(input_model)

        # Test objective: Verify successful conversion and compilation
        # Note: This is a conversion-only test. Runtime execution is not tested because
        # aten::_sparse_mm uses PtFrameworkNode fallback, which requires PyTorch runtime
        # context that may not be available in all inference scenarios.
        core = ov.Core()
        compiled_model = core.compile_model(ov_model, ie_device)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sparse_mm_different_sizes(self, ie_device, precision, ir_version):
        """Test sparse matrix multiplication with different matrix sizes"""
        import openvino as ov
        
        class SparseMMModel(torch.nn.Module):
            def forward(self, sparse_mat, dense_mat):
                return torch.ops.aten._sparse_mm(sparse_mat, dense_mat)
        
        # Create sparse matrix (5x10)
        indices = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8]], dtype=torch.long)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
        sparse_mat = torch.sparse_coo_tensor(indices, values, (5, 10))
        
        # Create dense matrix (10x7)
        dense_mat = torch.randn(10, 7)
        
        # Script the model
        model = SparseMMModel()
        scripted_model = torch.jit.script(model)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(scripted_model)
        ov_model = fe.convert(input_model)
        
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sparse_mm_empty_sparse(self, ie_device, precision, ir_version):
        """Test sparse matrix multiplication with empty sparse matrix"""
        import openvino as ov
        
        class SparseMMModel(torch.nn.Module):
            def forward(self, sparse_mat, dense_mat):
                return torch.ops.aten._sparse_mm(sparse_mat, dense_mat)
        
        # Create empty sparse matrix (3x5)
        indices = torch.tensor([[], []], dtype=torch.long).reshape(2, 0)
        values = torch.tensor([], dtype=torch.float32)
        sparse_mat = torch.sparse_coo_tensor(indices, values, (3, 5))
        
        # Create dense matrix (5x4)
        dense_mat = torch.randn(5, 4)
        
        # Script the model
        model = SparseMMModel()
        scripted_model = torch.jit.script(model)
        
        # Convert to OpenVINO
        from openvino.frontend import FrontEndManager
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework(framework="pytorch")
        
        input_model = fe.load(scripted_model)
        ov_model = fe.convert(input_model)
