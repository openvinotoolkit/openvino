# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestCosineSimilarity(PytorchLayerTest):

    def _prepare_input(self):
        x1 = self.random.randn(*self.x1_shape, dtype=self.dtype) * self.scale
        x2 = self.random.randn(*self.x2_shape, dtype=self.dtype) * self.scale
        if self.zero_x1:
            x1 = np.zeros_like(x1)
        return (x1, x2)

    def create_model(self, dim, eps):

        class aten_cosine_similarity(torch.nn.Module):
            def __init__(self, dim, eps):
                super().__init__()
                self.dim = dim
                self.eps = eps

            def forward(self, x1, x2):
                if self.dim is None:
                    return F.cosine_similarity(x1, x2)
                if self.eps is None:
                    return F.cosine_similarity(x1, x2, dim=self.dim)
                return F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)

        return aten_cosine_similarity(dim, eps), "aten::cosine_similarity"

    def _run(self, ie_device, precision, ir_version, x1_shape, x2_shape, dim, eps,
             dtype="float32", scale=1.0, zero_x1=False):
        self.x1_shape = x1_shape
        self.x2_shape = x2_shape
        self.dtype = dtype
        self.scale = scale
        self.zero_x1 = zero_x1
        kwargs = {}
        if dtype == "float16":
            # PyTorch rounds every intermediate to f16, so allow one f16 step of difference.
            kwargs["custom_eps"] = 1e-3
        self._test(*self.create_model(dim, eps), ie_device, precision, ir_version,
                   use_convert_model=True, fx_kind="aten.cosine_similarity", **kwargs)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize(("x1_shape", "x2_shape", "dim"), [
        ([4, 8], [4, 8], None),
        ([4, 8], [4, 8], 1),
        ([2, 3, 5], [2, 3, 5], 0),
        ([2, 3, 5], [2, 3, 5], -1),
        ([5], [5], 0),
        # x2 is broadcast along dim before the norms are taken
        ([4, 8], [4, 1], 1),
        ([2, 3, 5], [1, 3, 5], 1),
        ([2, 1, 5], [1, 4, 5], 2),
    ])
    @pytest.mark.parametrize("eps", [None, 1e-4])
    def test_cosine_similarity(self, x1_shape, x2_shape, dim, eps, ie_device, precision, ir_version):
        if dim is None and eps is not None:
            pytest.skip("eps is only passed together with dim")
        self._run(ie_device, precision, ir_version, x1_shape, x2_shape, dim, eps)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("eps", [None, 1e-4])
    def test_cosine_similarity_small_norm(self, eps, ie_device, precision, ir_version):
        # Norms around 3e-5 are below eps=1e-4, so each norm is clamped separately.
        self._run(ie_device, precision, ir_version, [4, 8], [4, 8], 1, eps, scale=1e-5)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_cosine_similarity_zero_vector(self, ie_device, precision, ir_version):
        self._run(ie_device, precision, ir_version, [4, 8], [4, 8], 1, None, zero_x1=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", ["float16", "float64"])
    def test_cosine_similarity_dtype(self, dtype, ie_device, precision, ir_version):
        self._run(ie_device, precision, ir_version, [4, 8], [4, 8], 1, None, dtype=dtype)
