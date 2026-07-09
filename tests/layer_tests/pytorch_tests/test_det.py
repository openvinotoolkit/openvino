# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import openvino as ov

from pytorch_layer_test_class import PytorchLayerTest


class TestDet(PytorchLayerTest):
    """aten::det / aten::linalg_det — batched matrix determinant.

    The PyTorch frontend computes the determinant of 3x3 matrices in closed
    form (cofactor expansion) -- the rigid-transform / Kabsch use case.
    Reference values come from PyTorch (torch.det / torch.linalg.det).
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
        # FP16 is too coarse for the cofactor products; validate the determinant in FP32.
        if precision == "FP16":
            pytest.skip("determinant closed form is validated in FP32")
        self._test(*self.create_model(variant), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape})


class TestDetGeneralSize(PytorchLayerTest):
    """aten::det / aten::linalg_det — determinant of a square matrix of arbitrary size N.

    Beyond the 3x3 closed form, the frontend computes a general NxN determinant via
    LU decomposition with partial pivoting (det = sign(P) * prod(diag(U))), exactly
    PyTorch's own algorithm. This works on the default dynamic trace path (the trailing
    matrix dims are dynamic there), so no static-shape pinning is needed. Reference
    values come from PyTorch (torch.det / torch.linalg.det).
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

        return aten_det(variant), "aten::linalg_det"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("variant", ["det", "linalg_det"])
    @pytest.mark.parametrize("input_shape", [
        [1, 1],          # 1x1
        [2, 2],          # 2x2
        [4, 4],          # 4x4, no batch
        [5, 5],          # 5x5, no batch
        [3, 4, 4],       # batch of 4x4
        [2, 3, 5, 5],    # multi-dim batch of 5x5
    ])
    def test_det_general_size(self, variant, input_shape, ie_device, precision, ir_version):
        # FP16 is too coarse for the LU products; validate the determinant in FP32.
        if precision == "FP16":
            pytest.skip("general-N determinant is validated in FP32")
        self._test(*self.create_model(variant), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape})


class TestDetNonSquareFailsGracefully(PytorchLayerTest):
    """A non-square determinant must fail loudly, never silently return a wrong value.

    The frontend supports any square NxN matrix (closed form for 1x1/2x2/3x3, LU
    decomposition otherwise). A non-square trailing pair is rejected: with statically
    known trailing dims the failure is raised at conversion time; when the trailing dims
    are dynamic the general LU path's [-1, N, N] flatten reshape makes it fail loudly at
    inference (op-labeled 'requires_square'). Either way a non-square matrix must not
    silently produce a wrong determinant.
    """

    def _det(self):
        class aten_det(torch.nn.Module):
            def forward(self, x):
                return torch.linalg.det(x)

        return aten_det()

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_static_nonsquare_fails_at_conversion(self, ie_device, precision, ir_version):
        # Static non-square trailing dims => the reshape element-count check fails while the
        # model is built (a RuntimeError, not OpConversionFailure). Trace on a square matrix
        # (torch rejects a non-square det at trace time) and force a non-square shape via input=.
        example = torch.randn(2, 4, 4, dtype=torch.float32)
        scripted = torch.jit.trace(self._det(), example)
        with pytest.raises(Exception):
            ov.convert_model(scripted, example_input=(example,),
                             input=[ov.PartialShape([2, 4, 5])])

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dynamic_nonsquare_fails_at_runtime(self, ie_device, precision, ir_version):
        # Dynamic trailing dims => conversion succeeds; a genuine non-square input must fail loudly at
        # inference (the LU [-1, N, N] flatten reshape cannot accept a non-square trailing pair), so
        # the determinant is never silently miscomputed.
        if ie_device != "CPU":
            pytest.skip("runtime reshape-guard failure is asserted on CPU")
        example = torch.randn(2, 4, 4, dtype=torch.float32)
        scripted = torch.jit.trace(self._det(), example)
        ov_model = ov.convert_model(scripted, example_input=(example,),
                                    input=[ov.PartialShape([2, -1, -1])])
        compiled = ov.Core().compile_model(ov_model, "CPU")
        bad = np.random.randn(2, 4, 5).astype(np.float32)
        with pytest.raises(Exception):
            compiled((bad,))

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_dynamic_same_numel_nonsquare_fails_at_runtime(self, ie_device, precision, ir_version):
        # [1, 9] has a square (3x3) element count: a [.., L, L] guard that pins to the actual last
        # dim would fold to a no-op, but the LU path's rank-changing [-1, N, N] flatten reshape
        # still fails loudly (1x9 has 9 elements, not a multiple of L*L for the resolved L).
        if ie_device != "CPU":
            pytest.skip("runtime reshape-guard failure is asserted on CPU")
        example = torch.randn(3, 3, dtype=torch.float32)
        scripted = torch.jit.trace(self._det(), example)
        ov_model = ov.convert_model(scripted, example_input=(example,),
                                    input=[ov.PartialShape([-1, -1])])
        compiled = ov.Core().compile_model(ov_model, "CPU")
        bad = np.arange(9, dtype=np.float32).reshape(1, 9)
        with pytest.raises(Exception) as exc_info:
            compiled((bad,))
        assert "requires_square" in str(exc_info.value)
