# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestTensordot(PytorchLayerTest):
    def _prepare_input(self, shape_a, shape_b):
        return (
            self.random.randn(*shape_a, dtype=np.float32),
            self.random.randn(*shape_b, dtype=np.float32)
        )

    def create_model(self, dims, use_out=False):
        class TensordotModel(torch.nn.Module):
            def __init__(self, dims, use_out=False):
                super().__init__()
                self.dims = dims
                self.use_out = use_out

            def forward(self, a, b):
                if self.use_out:
                    # Compute a reference result to infer the output shape, then
                    # allocate an output tensor and use the `out` argument.
                    ref = torch.tensordot(a, b, dims=self.dims)
                    out = torch.empty_like(ref)
                    return torch.tensordot(a, b, dims=self.dims, out=out)
                return torch.tensordot(a, b, dims=self.dims)

        return TensordotModel(dims, use_out), None, "aten::tensordot"

    @pytest.mark.parametrize("shape_a, shape_b, dims, use_out", [
        ((2, 3), (3, 2), 1, False),
        ((4, 5), (5, 4), 1, False),
        ((2, 3, 4), (4, 3, 2), 2, False),
        ((2, 3, 4), (3, 4, 2), ([1, 2], [0, 1]), False),  # Tuple of lists case
        ((2, 3, 4), (4, 3, 2), ([-1], [0]), False),        # Negative axis in dims
        ((2, 3), (3, 2), 1, True),                         # Use `out` argument
        # Non-trivial permutation: a_remain is NOT a leading prefix, so the
        # transposed shape differs from the original.  This exercises the
        # shape_source / input split in reshape_to_2d.
        ((2, 3, 4), (5, 3, 6), ([1], [1]), False),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tensordot(self, shape_a, shape_b, dims, use_out, ie_device, precision, ir_version):
        self._test(*self.create_model(dims, use_out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"shape_a": shape_a, "shape_b": shape_b})
