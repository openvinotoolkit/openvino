# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest

try:
    import torch_scatter
except ImportError:
    torch_scatter = None

class TestSegmentCSR(PytorchLayerTest):
    def _prepare_input(self, src_shape, indptr_shape, dtype):
        src = np.random.randn(*src_shape).astype(dtype)
        # Generate valid indptr
        # indptr must be monotonic increasing, start with 0, end with src.shape[indptr.ndim-1]
        k = len(indptr_shape) - 1
        dim_size = src_shape[k]

        # We need to generate random segments that sum up to dim_size
        # But for simplicity, let's just create some random sorted indices
        # indptr has length indptr_shape[-1]
        num_segments = indptr_shape[-1] - 1

        # Make sure indptr matches batch dimensions if any
        # If indptr has batch dims, we need to generate it carefully
        # For this test, let's assume simple cases or implement logic

        indptr = np.zeros(indptr_shape, dtype="int64")

        # Helper to fill 1D indptr
        def fill_indptr(arr, limit):
            n = len(arr)
            # generate n-1 cut points in [0, limit]
            cuts = np.sort(np.random.choice(limit + 1, n-2, replace=True))
            arr[0] = 0
            arr[1:-1] = cuts
            arr[-1] = limit

        if len(indptr_shape) == 1:
            fill_indptr(indptr, dim_size)
        else:
             # Flatten batch dims
             flat_indptr = indptr.reshape(-1, indptr_shape[-1])
             for i in range(flat_indptr.shape[0]):
                 fill_indptr(flat_indptr[i], dim_size)

        return src, indptr

    def create_model(self, reduce):
        class SegmentCSRModel(torch.nn.Module):
            def __init__(self, reduce):
                super().__init__()
                self.reduce = reduce

            def forward(self, src, indptr):
                return torch_scatter.segment_csr(src, indptr, reduce=self.reduce)

        return SegmentCSRModel(reduce), None, "torch_scatter::segment_mean_csr"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("src_shape", [[10, 6, 64], [10, 20]])
    @pytest.mark.parametrize("indptr_shape", [[4], [10, 5]])
    @pytest.mark.parametrize("dtype", ["float32"])
    @pytest.mark.parametrize("reduce", ["mean"])
    def test_segment_csr(self, src_shape, indptr_shape, dtype, reduce, ie_device, precision, ir_version):
        if torch_scatter is None:
            pytest.skip("torch_scatter is not installed")

        # Check compatibility of shapes
        # indptr.ndim must be <= src.ndim
        if len(indptr_shape) > len(src_shape):
            return # Invalid config

        # If broadcasting indptr, batch dims must match
        # src: [10, 6, 64], indptr: [10, 5]. k = 2-1 = 1. src[1] is 6. indptr last dim is 5 (4 segments).
        # indptr [10, 5] implies it segments dimension 1.
        # batch dims: src[0]=10, indptr[0]=10. Match.

        # If src: [10, 6, 64], indptr: [4]. k = 1-1 = 0.
        # This means we segment dimension 0. src[0] is 10. indptr last dim is 4 (3 segments).

        self._test(
            *self.create_model(reduce),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "src_shape": src_shape,
                "indptr_shape": indptr_shape,
                "dtype": dtype
            },
            trace_model=True
        )
