# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import openvino as ov

from pytorch_layer_test_class import PytorchLayerTest


class TestSVDReconstruction(PytorchLayerTest):
    """aten::svd — batched 3x3 singular value decomposition.

    The frontend implements a one-sided Jacobi SVD for 3x3 matrices. Singular
    vectors are only defined up to sign, so the model wrapper below returns the
    sign-invariant reconstruction U @ diag(S) @ V^T (which must equal the input)
    and the sorted singular values. Reference values come from PyTorch
    (torch.svd).
    """

    def _prepare_input(self, input_shape, rank_deficient=False):
        x = self.random.randn(*input_shape).astype(np.float32)
        if rank_deficient:
            # Make the last column equal the second => mathematically rank<=2,
            # which is the SAM-6D Weighted-Procrustes (3-point) regime.
            x[..., :, -1] = x[..., :, -2]
        return (x,)

    def create_model(self):
        class aten_svd_recon(torch.nn.Module):
            def forward(self, x):
                u, s, v = torch.svd(x)
                recon = torch.matmul(u * s.unsqueeze(-2), v.transpose(-2, -1))
                # Return reconstruction (sign-invariant) and sorted singular values.
                return recon, s

        return aten_svd_recon(), "aten::svd"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [
        [3, 3],          # single 3x3
        [1, 3, 3],       # batch of one
        [8, 3, 3],       # batch
        [2, 4, 3, 3],    # multi-dim batch
    ])
    @pytest.mark.parametrize("rank_deficient", [False, True])
    def test_svd_reconstruction(self, input_shape, rank_deficient,
                                ie_device, precision, ir_version):
        # The Jacobi rotations and singular values are computed in FP32; FP16 is
        # too coarse (it forms products of matrix entries). Keep FP32.
        if precision == "FP16":
            pytest.skip("3x3 SVD is validated in FP32")
        # The 3x3 SVD requires statically 3x3 trailing dims; disable the harness's
        # default fully-dynamic-rank tracing.
        self._test(*self.create_model(), ie_device, precision, ir_version,
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


class TestSVDNonSquareFailsGracefully(PytorchLayerTest):
    """A non-3x3 SVD must fail loudly, never silently return a wrong decomposition.

    The frontend only implements the 3x3 SVD. ensure_trailing_square (utils.cpp)
    rejects any other size: with statically-known trailing dims the op-labeled
    error is raised at conversion time; when the trailing dims are dynamic a
    runtime reshape guard makes it fail at inference. Either way a 4x4 matrix must
    not silently produce a (wrong) 3x3 decomposition.
    """

    def _svd(self):
        class aten_svd(torch.nn.Module):
            def forward(self, x):
                u, s, v = torch.svd(x)
                return u, s, v

        return aten_svd()

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_static_4x4_fails_at_conversion(self, ie_device, precision, ir_version):
        # Static trailing dims => the conversion-time check in ensure_trailing_square fires.
        example = torch.randn(2, 4, 4, dtype=torch.float32)
        scripted = torch.jit.trace(self._svd(), example)
        # Not OpConversionFailure: the 3x3 guard trips core Reshape shape-inference while the model
        # is built, surfaced as a RuntimeError. Assert the op-labeled guard node so an unrelated
        # failure (e.g. a tracing error) cannot green this test.
        with pytest.raises(Exception) as exc_info:
            ov.convert_model(scripted, example_input=(example,),
                             input=[ov.PartialShape([2, 4, 4])])
        assert "requires_3x3" in str(exc_info.value)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dynamic_4x4_fails_at_runtime(self, ie_device, precision, ir_version):
        # Dynamic trailing dims => conversion succeeds with the runtime reshape guard,
        # which then fails loudly at inference for a genuine 4x4 input.
        if ie_device != "CPU":
            pytest.skip("runtime reshape-guard failure is asserted on CPU")
        example = torch.randn(2, 4, 4, dtype=torch.float32)
        scripted = torch.jit.trace(self._svd(), example)
        ov_model = ov.convert_model(scripted, example_input=(example,),
                                    input=[ov.PartialShape([2, -1, -1])])
        compiled = ov.Core().compile_model(ov_model, "CPU")
        # Runtime failure in the CPU plugin (a RuntimeError, not OpConversionFailure). Assert the
        # op-labeled guard node so an unrelated failure cannot green this test.
        with pytest.raises(Exception) as exc_info:
            compiled((example.numpy(),))
        assert "requires_3x3" in str(exc_info.value)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dynamic_same_numel_nonsquare_fails_at_runtime(self, ie_device, precision, ir_version):
        # [1, 9] has 3x3's element count: a single [n,n] reshape would accept it; the per-axis guard must raise.
        if ie_device != "CPU":
            pytest.skip("runtime reshape-guard failure is asserted on CPU")
        example = torch.randn(3, 3, dtype=torch.float32)
        scripted = torch.jit.trace(self._svd(), example)
        ov_model = ov.convert_model(scripted, example_input=(example,),
                                    input=[ov.PartialShape([-1, -1])])
        compiled = ov.Core().compile_model(ov_model, "CPU")
        bad = np.arange(9, dtype=np.float32).reshape(1, 9)
        with pytest.raises(Exception) as exc_info:
            compiled((bad,))
        assert "requires_3x3" in str(exc_info.value)


class TestSVDComputeUvFalseFailsGracefully(PytorchLayerTest):
    """aten::svd with compute_uv=False must fail at conversion, not silently return non-zero U/V.

    PyTorch returns zero-filled U/V when compute_uv=False; this frontend always produces real
    U/V, so it rejects a constant compute_uv=False at conversion time. (A non-constant compute_uv
    is likewise rejected in the translator, but Python bool args const-fold into prim::Constant
    under torch.jit.trace, so that branch is not reachable from a layer test.)
    """

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_compute_uv_false_fails_at_conversion(self, ie_device, precision, ir_version):
        class aten_svd_no_uv(torch.nn.Module):
            def forward(self, x):
                _, s, _ = torch.svd(x, compute_uv=False)
                return s

        example = torch.randn(2, 3, 3, dtype=torch.float32)
        scripted = torch.jit.trace(aten_svd_no_uv(), example)
        # The translator's PYTORCH_OP_CONVERSION_CHECK surfaces as OpConversionFailure; asserting
        # the exact type keeps an unrelated failure from greening this test.
        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(example,),
                             input=[ov.PartialShape([2, 3, 3])])
