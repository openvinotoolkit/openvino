# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class aten_sparse_mm(torch.nn.Module):
    def __init__(self, sparse_shape, indices, values):
        super().__init__()
        self.sparse_shape = sparse_shape
        self.indices = indices
        self.values = values

    def forward(self, dense_mat):
        sparse_mat = torch.sparse_coo_tensor(self.indices, self.values, self.sparse_shape)
        return torch.sparse.mm(sparse_mat, dense_mat)

class TestSparseMM(PytorchLayerTest):
    def _prepare_input(self, dense_shape):
        return (np.random.randn(*dense_shape).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    # torch.export fails with "NotImplementedError: Cannot access storage of SparseTensorImpl"
    # This is an upstream PyTorch limitation for sparse tensors in FakeTensor.
    @pytest.mark.parametrize("sparse_shape, dense_shape", [
        ((3, 3), (3, 2)),
        ((4, 5), (5, 2)),
        ((10, 10), (10, 10)),
    ])
    def test_sparse_mm(self, sparse_shape, dense_shape, ie_device, precision, ir_version):
        # Create a random sparse matrix
        num_elements = sparse_shape[0] # Just some elements
        # Ensure unique indices to avoid collision (PyTorch sums, ScatterNDUpdate overwrites)
        total_elements = sparse_shape[0] * sparse_shape[1]
        flat_indices = np.random.choice(total_elements, num_elements, replace=False)
        row_indices = flat_indices // sparse_shape[1]
        col_indices = flat_indices % sparse_shape[1]
        
        indices = torch.tensor([
            row_indices,
            col_indices
        ], dtype=torch.int64)
        values = torch.randn(num_elements, dtype=torch.float32)
        
        self._test(aten_sparse_mm(sparse_shape, indices, values), 
                   None, "aten::_sparse_mm", 
                   ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"dense_shape": dense_shape},
                   trace_model=True) # sparse_mm requires tracing usually as scripting often fails with sparse construction inside

class aten_sparse_mm_sxs(torch.nn.Module):
    def __init__(self, sparse_shape1, indices1, values1, sparse_shape2, indices2, values2):
        super().__init__()
        self.sparse_shape1 = sparse_shape1
        self.indices1 = indices1
        self.values1 = values1
        self.sparse_shape2 = sparse_shape2
        self.indices2 = indices2
        self.values2 = values2

    def forward(self):
        sparse_mat1 = torch.sparse_coo_tensor(self.indices1, self.values1, self.sparse_shape1)
        sparse_mat2 = torch.sparse_coo_tensor(self.indices2, self.values2, self.sparse_shape2)
        result = torch.sparse.mm(sparse_mat1, sparse_mat2)
        return result.to_dense()  # Convert to dense for comparison

class TestSparseMMSxS(PytorchLayerTest):
    def _prepare_input(self):
        return ()  # No runtime inputs needed

    @pytest.mark.nightly
    @pytest.mark.precommit
    # torch.export fails with "NotImplementedError: Cannot access storage of SparseTensorImpl"
    # This is an upstream PyTorch limitation for sparse tensors in FakeTensor.
    @pytest.mark.parametrize("shape1, shape2", [
        ((3, 4), (4, 2)),
        ((5, 5), (5, 3)),
        ((10, 8), (8, 10)),
    ])
    def test_sparse_mm_sxs(self, shape1, shape2, ie_device, precision, ir_version):
        # Create first sparse matrix
        num_elements1 = shape1[0]
        total_elements1 = shape1[0] * shape1[1]
        flat_indices1 = np.random.choice(total_elements1, num_elements1, replace=False)
        row_indices1 = flat_indices1 // shape1[1]
        col_indices1 = flat_indices1 % shape1[1]
        indices1 = torch.tensor([row_indices1, col_indices1], dtype=torch.int64)
        values1 = torch.randn(num_elements1, dtype=torch.float32)
        
        # Create second sparse matrix
        num_elements2 = shape2[0]
        total_elements2 = shape2[0] * shape2[1]
        flat_indices2 = np.random.choice(total_elements2, num_elements2, replace=False)
        row_indices2 = flat_indices2 // shape2[1]
        col_indices2 = flat_indices2 % shape2[1]
        indices2 = torch.tensor([row_indices2, col_indices2], dtype=torch.int64)
        values2 = torch.randn(num_elements2, dtype=torch.float32)
        
        self._test(
            aten_sparse_mm_sxs(shape1, indices1, values1, shape2, indices2, values2),
            None,
            "aten::_sparse_mm",
            ie_device,
            precision,
            ir_version,
            trace_model=True
        )
