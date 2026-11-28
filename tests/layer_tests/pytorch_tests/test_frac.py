# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestFrac(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(self.data, dtype=self.dtype),)

    def create_model(self):
        class aten_frac(torch.nn.Module):
            def forward(self, x):
                return torch.frac(x)

        return aten_frac(), "aten::frac"

    @pytest.mark.parametrize("dtype,data", [
        # mixed positive and negative values, 1-D
        (np.float32, [1.5, -1.5, 2.7, -2.7, 0.25, -0.75]),
        (np.float64, [1.5, -1.5, 2.7, -2.7, 0.25, -0.75]),
        # exact integers: fractional part is 0
        (np.float32, [0.0, 1.0, -1.0, 10.0, -7.0]),
        (np.float64, [0.0, 1.0, -1.0, 10.0, -7.0]),
        # 0-D scalar
        (np.float32, 3.14),
        (np.float64, 3.14),
        # 2-D
        (np.float32, [[1.1, -2.3], [3.0, -4.9]]),
        (np.float64, [[1.1, -2.3], [3.0, -4.9]]),
        # 3-D
        (np.float32, [[[0.5, -0.5], [1.25, -2.75]]]),
        (np.float64, [[[0.5, -0.5], [1.25, -2.75]]]),
        # 4-D
        (np.float32, [[[[1.5, -0.25], [2.0, -3.75]]]]),
        (np.float64, [[[[1.5, -0.25], [2.0, -3.75]]]]),
        # large magnitudes; torch is the oracle for lost fractional bits
        (np.float32, [1048576.5, -1048576.25, 1.0e8, -1.0e8]),
        (np.float64, [1048576.5, -1048576.25, 1.0e8, -1.0e8]),
        # inf / -inf produce NaN; NaN is preserved. float32 only: with
        # TEST_PRECISION=FP32 the CPU plugin evaluates f64 as f32 and saturates
        # inf to FLT_MAX, so frac(inf) becomes 0 instead of NaN.
        (np.float32, [np.inf, -np.inf, np.nan, 1.5, -2.25]),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_frac(self, dtype, data, ie_device, precision, ir_version):
        self.dtype = dtype
        self.data = data
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestFracInplace(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([1.5, -2.3, 4.0, -0.75, 0.0], dtype=np.float32),)

    def create_model(self):
        class aten_frac_(torch.nn.Module):
            def forward(self, x):
                return x.frac_()

        return aten_frac_(), "aten::frac_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_frac_inplace(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestFracOut(PytorchLayerTest):
    def _prepare_input(self):
        x = np.array([1.5, -2.3, 4.0, -0.75, 0.0], dtype=np.float32)
        return (x, np.zeros_like(x))

    def create_model(self):
        class aten_frac_out(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_op = PytorchLayerTest.out_variant(torch.frac, torch.ops.aten.frac.out)

            def forward(self, x, out):
                return self.out_op(x, out=out), out

        return aten_frac_out(), "aten::frac"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_frac_out(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
