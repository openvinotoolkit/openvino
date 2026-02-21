# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest

try:
    from torch_scatter import segment_csr
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    TORCH_SCATTER_AVAILABLE = False


@pytest.mark.skipif(not TORCH_SCATTER_AVAILABLE, reason="torch_scatter is not installed")
@pytest.mark.skipif(
    PytorchLayerTest.use_torch_export() or PytorchLayerTest.use_torch_compile_backend(),
    reason="torch_scatter::segment_mean_csr is only supported for TorchScript",
)
class TestSegmentMeanCSR(PytorchLayerTest):
    def _prepare_input(self, shape, dtype):
        # Use arange for easier debugging of segment boundaries
        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        return (data,)

    def create_model(self, indptr):
        class SegmentMeanCSR(torch.nn.Module):
            def __init__(self, indptr):
                super().__init__()
                self.indptr = indptr

            def forward(self, src):
                # Use segment_csr with reduce="mean" as shown in the issue example
                return segment_csr(src, self.indptr, reduce="mean")  # type: ignore[possibly-undefined]

        ref_net = None
        # segment_csr with reduce="mean" emits torch_scatter::segment_mean_csr in TorchScript
        return SegmentMeanCSR(indptr), ref_net, "torch_scatter::segment_mean_csr"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "src_shape,indptr",
        [
            # 2D indptr with shape (1, n) - segments on dim 1 (axis = 2 - 1 = 1)
            ((10, 6, 64), torch.tensor([[0, 2, 5, 6]], dtype=torch.int64)),
            # 1D indptr - segments on dim 0 (axis = 1 - 1 = 0)
            ((6, 4), torch.tensor([0, 3, 6], dtype=torch.int64)),
            # Single element segments
            ((4, 8), torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)),
            # Different batch size
            ((5, 10, 32), torch.tensor([[0, 3, 7, 10]], dtype=torch.int64)),
        ],
    )
    def test_segment_mean_csr(self, src_shape, indptr, ie_device, precision, ir_version):
        self._test(
            *self.create_model(indptr),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"shape": src_shape, "dtype": np.float32},
            freeze_model=True,
        )

