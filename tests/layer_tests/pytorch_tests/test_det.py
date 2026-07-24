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


class TestDetNonSquareFailsGracefully(PytorchLayerTest):
    """A non-3x3 determinant must fail loudly, never silently return a wrong value.

    The frontend only implements the 3x3 determinant. ensure_trailing_square
    (utils.cpp) rejects any other size: with statically-known trailing dims the
    op-labeled error is raised at conversion time; when the trailing dims are
    dynamic a runtime reshape guard makes it fail at inference. Either way a 4x4
    matrix must not silently produce a (wrong) 3x3 determinant.
    """

    def _det(self):
        class aten_det(torch.nn.Module):
            def forward(self, x):
                return torch.linalg.det(x)

        return aten_det()

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_static_4x4_fails_at_conversion(self, ie_device, precision, ir_version):
        # Static trailing dims => the conversion-time check in ensure_trailing_square fires.
        example = torch.randn(2, 4, 4, dtype=torch.float32)
        scripted = torch.jit.trace(self._det(), example)
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
        scripted = torch.jit.trace(self._det(), example)
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
        scripted = torch.jit.trace(self._det(), example)
        ov_model = ov.convert_model(scripted, example_input=(example,),
                                    input=[ov.PartialShape([-1, -1])])
        compiled = ov.Core().compile_model(ov_model, "CPU")
        bad = np.arange(9, dtype=np.float32).reshape(1, 9)
        with pytest.raises(Exception) as exc_info:
            compiled((bad,))
        assert "requires_3x3" in str(exc_info.value)
