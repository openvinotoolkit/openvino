# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize("input_shape_rhs", [
    [2, 5, 3, 4],
    [1, 5, 3, 4],
    [1]
])
class TestHypot(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(2, 5, 3, 4), self.input_rhs)

    def create_model(self):

        class aten_hypot(torch.nn.Module):
            def forward(self, lhs, rhs):
                return torch.hypot(lhs, rhs)

        return aten_hypot(), "aten::hypot"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_hypot(self, ie_device, precision, ir_version, input_shape_rhs):
        self.input_rhs = self.random.randn(*input_shape_rhs)
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   use_convert_model=True, fx_kind="aten.hypot")


class TestHypotTypes(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(*self.lhs_shape, dtype=self.lhs_type),
                self.random.randn(*self.rhs_shape, dtype=self.rhs_type))

    def create_model(self, lhs_type, rhs_type):

        class aten_hypot(torch.nn.Module):
            def __init__(self, lhs_type, rhs_type):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type

            def forward(self, lhs, rhs):
                return torch.hypot(lhs.to(self.lhs_type), rhs.to(self.rhs_type))

        return aten_hypot(lhs_type, rhs_type), "aten::hypot"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.float16, torch.float32],
                              [torch.float32, torch.float64],
                              [torch.float16, torch.float64],
                              [torch.float32, torch.float32]])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], [1, 3]),
                                                          ([3, 2, 3], [2, 3])])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_hypot_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, rhs_type),
                   ie_device, precision, ir_version, freeze_model=False, trace_model=True, fx_kind="aten.hypot")


class TestHypotValues(PytorchLayerTest):
    # Pythagorean triples plus negative and zero legs give exact, easy-to-verify results:
    # (3,4)->5, (5,12)->13, (8,15)->17, (-6,8)->10, (0,0)->0.
    def _prepare_input(self):
        return (np.array([3.0, 5.0, 8.0, -6.0, 0.0], dtype=np.float32),
                np.array([4.0, 12.0, 15.0, 8.0, 0.0], dtype=np.float32))

    def create_model(self):

        class aten_hypot(torch.nn.Module):
            def forward(self, lhs, rhs):
                return torch.hypot(lhs, rhs)

        return aten_hypot(), "aten::hypot"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_hypot_values(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.hypot")


class TestHypotExtremeValues(PytorchLayerTest):
    # Inputs whose squares overflow to inf or underflow to 0 in f32, while hypot itself is
    # representable. The result is divided by max(|input|, |other|) so every finite output is
    # in [1, sqrt(2)] and the absolute tolerance of the comparison cannot hide an inf or a 0.
    def _prepare_input(self):
        lhs = np.array(self.lhs, dtype=np.float32)
        rhs = np.array(self.rhs, dtype=np.float32)
        norm = np.maximum(np.abs(lhs), np.abs(rhs))
        norm = np.where(np.isfinite(norm) & (norm > 0), norm, 1).astype(np.float32)
        return (lhs, rhs, norm)

    def create_model(self):

        class aten_hypot(torch.nn.Module):
            def forward(self, lhs, rhs, norm):
                return torch.hypot(lhs, rhs) / norm

        return aten_hypot(), "aten::hypot"

    @pytest.mark.parametrize(("lhs", "rhs"), [
        # squares overflow f32
        ([1e20, 3e19, -1e30, 1e38, 1e38], [1e20, 4e19, 1e30, 1e38, 1.0]),
        # squares underflow to 0 in f32
        ([1e-25, 3e-30, -1e-37, 1e-30], [1e-25, 4e-30, 1e-37, 0.0]),
        # inf dominates nan, nan propagates otherwise
        ([np.inf, -np.inf, np.nan, np.nan, np.inf, 3.0, 0.0], [np.nan, 5.0, -np.inf, 2.0, np.inf, np.nan, 0.0]),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_hypot_extreme_values(self, ie_device, precision, ir_version, lhs, rhs):
        if precision == "FP16":
            pytest.skip("Values exceed the f16 range")
        self.lhs = lhs
        self.rhs = rhs
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.hypot")


class TestHypotF16Range(PytorchLayerTest):
    # In f16 (max 65504) the squares of inputs above ~256 overflow, though hypot itself fits.
    def _prepare_input(self):
        return (np.array([300.0, -1000.0, 255.0, 40000.0, 3.0], dtype=np.float32),
                np.array([400.0, 0.5, 256.0, 30000.0, 4.0], dtype=np.float32))

    def create_model(self):

        class aten_hypot(torch.nn.Module):
            def forward(self, lhs, rhs):
                return torch.hypot(lhs.to(torch.float16), rhs.to(torch.float16))

        return aten_hypot(), "aten::hypot"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_hypot_f16_range(self, ie_device, precision, ir_version):
        # f16 resolution is ~1e-3 relative, results may differ from torch by an ulp.
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   freeze_model=False, trace_model=True, fx_kind="aten.hypot", custom_eps=1e-2)
