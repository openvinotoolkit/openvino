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
