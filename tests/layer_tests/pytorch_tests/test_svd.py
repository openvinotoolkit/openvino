# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSVDReconstruction(PytorchLayerTest):
    """aten::svd / aten::linalg_svd — batched 3x3 singular value decomposition.

    The frontend implements a one-sided Jacobi SVD for 3x3 matrices. Singular
    vectors are only defined up to sign, so the model wrappers below return the
    sign-invariant reconstruction U @ diag(S) @ V^T (which must equal the input)
    and the sorted singular values. Reference values come from PyTorch
    (torch.svd / torch.linalg.svd).
    """

    def _prepare_input(self, input_shape, rank_deficient=False):
        x = self.random.randn(*input_shape).astype(np.float32)
        if rank_deficient:
            # Make the last column equal the second => mathematically rank<=2,
            # which is the SAM-6D Weighted-Procrustes (3-point) regime.
            x[..., :, -1] = x[..., :, -2]
        return (x,)

    def create_model(self, variant):
        class aten_svd_recon(torch.nn.Module):
            def __init__(self, variant):
                super().__init__()
                self.variant = variant

            def forward(self, x):
                if self.variant == "linalg_svd":
                    u, s, vh = torch.linalg.svd(x)
                    recon = torch.matmul(u * s.unsqueeze(-2), vh)
                else:
                    u, s, v = torch.svd(x)
                    recon = torch.matmul(u * s.unsqueeze(-2), v.transpose(-2, -1))
                # Return reconstruction (sign-invariant) and sorted singular values.
                return recon, s

        op = "aten::linalg_svd" if variant == "linalg_svd" else "aten::svd"
        return aten_svd_recon(variant), op

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("variant", ["svd", "linalg_svd"])
    @pytest.mark.parametrize("input_shape", [
        [3, 3],          # single 3x3
        [1, 3, 3],       # batch of one
        [8, 3, 3],       # batch
        [2, 4, 3, 3],    # multi-dim batch
    ])
    @pytest.mark.parametrize("rank_deficient", [False, True])
    def test_svd_reconstruction(self, variant, input_shape, rank_deficient,
                                ie_device, precision, ir_version):
        # The Jacobi rotations and singular values are computed in FP32; FP16 is
        # too coarse (it forms products of matrix entries). Keep FP32.
        if precision == "FP16":
            pytest.skip("3x3 SVD is validated in FP32")
        # The 3x3 SVD requires statically 3x3 trailing dims; disable the harness's
        # default fully-dynamic-rank tracing.
        self._test(*self.create_model(variant), ie_device, precision, ir_version,
                   dynamic_shapes=False,
                   kwargs_to_prepare_input={"input_shape": input_shape,
                                            "rank_deficient": rank_deficient})


class TestSVDProcrustes(PytorchLayerTest):
    """End-to-end Weighted-Procrustes (Kabsch) rotation built from svd + det.

    This mirrors the exact SAM-6D usage: R = V @ diag(1,1,sign(det(V U^T))) @ U^T
    on a 3x3 cross-covariance matrix. The recovered rotation is invariant to the
    singular-vector sign ambiguity, so it compares cleanly against PyTorch.
    """

    def _prepare_input(self, batch):
        # Build a 3x3 cross-covariance from 3 random point correspondences (the
        # rank<=2 regime that SAM-6D's coarse matching produces).
        src = self.random.randn(batch, 3, 3).astype(np.float32)
        ref = self.random.randn(batch, 3, 3).astype(np.float32)
        return (src, ref)

    def create_model(self):
        class kabsch(torch.nn.Module):
            def forward(self, src, ref):
                src_c = src - src.mean(dim=1, keepdim=True)
                ref_c = ref - ref.mean(dim=1, keepdim=True)
                h = src_c.transpose(1, 2) @ ref_c
                u, _, v = torch.svd(h)
                ut = u.transpose(1, 2)
                eye = torch.eye(3).unsqueeze(0).repeat(src.shape[0], 1, 1)
                eye[:, -1, -1] = torch.sign(torch.det(v @ ut))
                r = v @ eye @ ut
                return r

        # Exercises both aten::svd and aten::det (torch.det lowers to
        # aten::linalg_det in TorchScript for current PyTorch).
        return kabsch(), ["aten::svd", "aten::linalg_det"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("batch", [1, 16])
    def test_procrustes(self, batch, ie_device, precision, ir_version):
        if precision == "FP16":
            pytest.skip("Procrustes (svd+det) is validated in FP32")
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   dynamic_shapes=False,
                   kwargs_to_prepare_input={"batch": batch})
