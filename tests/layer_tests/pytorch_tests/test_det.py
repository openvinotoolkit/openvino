# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestDet(PytorchLayerTest):
    """aten::det / aten::linalg_det — batched matrix determinant.

    The PyTorch frontend computes the determinant of small (1x1, 2x2, 3x3)
    matrices in closed form (cofactor expansion). Reference values come from
    PyTorch (torch.det / torch.linalg.det).
    """

    def _prepare_input(self, input_shape):
        return (self.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, variant):
        class aten_det(torch.nn.Module):
            def __init__(self, variant):
                super().__init__()
                self.variant = variant

            def forward(self, x):
                if self.variant == "linalg_det":
                    return torch.linalg.det(x)
                return torch.det(x)

        # Both torch.det and torch.linalg.det lower to aten::linalg_det in
        # TorchScript for current PyTorch.
        return aten_det(variant), "aten::linalg_det"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("variant", ["det", "linalg_det"])
    @pytest.mark.parametrize("input_shape", [
        [3, 3],          # single matrix, no batch
        [1, 3, 3],       # batch of one 3x3
        [5, 3, 3],       # batch of 3x3
        [2, 4, 3, 3],    # multi-dim batch of 3x3
    ])
    def test_det(self, variant, input_shape, ie_device, precision, ir_version):
        # FP16 squares element magnitudes in the cofactor products and is too
        # coarse for a meaningful determinant comparison; keep FP32.
        if precision == "FP16":
            pytest.skip("determinant closed form is validated in FP32")
        # The PyTorch frontend presents the input with a dynamic shape at
        # conversion time, so the translator decomposes the supported 3x3 case;
        # these tests therefore cover 3x3 matrices (the rigid-transform use case).
        self._test(*self.create_model(variant), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape})
