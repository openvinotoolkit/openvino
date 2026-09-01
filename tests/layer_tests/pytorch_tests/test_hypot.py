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
