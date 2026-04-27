# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestSparseMM(PytorchLayerTest):
    def _prepare_input(self, dense_shape):
        rng = np.random.default_rng(42)
        return (rng.standard_normal(dense_shape).astype(np.float32),)

    def create_model(self, sparse_shape, indices, values):
        class aten_sparse_mm(torch.nn.Module):
            def __init__(self, sparse_shape, indices, values):
                super().__init__()
                self.sparse_shape = sparse_shape
                self.indices = indices
                self.values = values

            def forward(self, dense_mat):
                sparse_mat = torch.sparse_coo_tensor(self.indices, self.values, self.sparse_shape)
                return torch.sparse.mm(sparse_mat, dense_mat)

        return aten_sparse_mm(sparse_shape, indices, values), "aten::_sparse_mm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("sparse_shape", "dense_shape", "sparsity_mode"), [
        skip_if_export((3, 3), (3, 2), "empty"),
        skip_if_export((3, 3), (3, 2), "very_sparse"),
        skip_if_export((3, 3), (3, 2), "per_row"),
        skip_if_export((4, 5), (5, 2), "multi_per_row"),
        skip_if_export((10, 10), (10, 10), "multi_per_row"),
    ])
    def test_sparse_mm(self, sparse_shape, dense_shape, sparsity_mode, ie_device, precision, ir_version):
        rng = np.random.default_rng(42)
        total_elements = sparse_shape[0] * sparse_shape[1]
        
        if sparsity_mode == "empty":
            num_elements = 0
        elif sparsity_mode == "very_sparse":
            num_elements = 1
        elif sparsity_mode == "per_row":
            num_elements = min(sparse_shape[0], total_elements)
        elif sparsity_mode == "multi_per_row":
            num_elements = min(sparse_shape[0] * 2, total_elements)
        else:
            raise ValueError(f"Unknown sparsity_mode: {sparsity_mode}")

        flat_indices = rng.choice(total_elements, num_elements, replace=False)
        row_indices = flat_indices // sparse_shape[1]
        col_indices = flat_indices % sparse_shape[1]

        torch.manual_seed(42)
        indices = torch.tensor(np.array([row_indices, col_indices]), dtype=torch.int64)
        values = torch.randn(num_elements, dtype=torch.float32)

        self._test(
            *self.create_model(sparse_shape, indices, values),
            ie_device, precision, ir_version,
            kwargs_to_prepare_input={"dense_shape": dense_shape},
            trace_model=True,
        )