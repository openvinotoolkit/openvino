# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

@pytest.mark.parametrize("input_shape_rhs", [
    [2, 5, 3, 4],
    [1, 5, 3, 4],
    [1]
])
class TestAtan2(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(2, 5, 3, 4), self.input_rhs)

    def create_model(self):

        class aten_atan2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, lhs, rhs):
                return torch.arctan2(lhs, rhs)


        return aten_atan2(), "aten::atan2"
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_atan2(self, ie_device, precision, ir_version, input_shape_rhs):
        self.input_rhs = self.random.randn(*input_shape_rhs)
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True, fx_kind="aten.arctan2")

class TestAtan2Types(PytorchLayerTest):

    def _prepare_input(self):
        return (self.random.randn(*self.lhs_shape, dtype=self.lhs_type),
            self.random.randn(*self.rhs_shape, dtype=self.rhs_type))

    def create_model(self, lhs_type, rhs_type):

        class aten_atan2(torch.nn.Module):
            def __init__(self, lhs_type, rhs_type):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type

            def forward(self, lhs, rhs):
                return torch.arctan2(lhs.to(self.lhs_type), rhs.to(self.rhs_type))


        return aten_atan2(lhs_type, rhs_type), "aten::atan2"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.int, torch.float32],
                              [torch.int, torch.float64],
                              [torch.float32, torch.float64],
                              [torch.int64, torch.float32]
                              ])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], [1, 3]),
                                                          ([3, 2, 3], [2, 3]),
                                                          ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_atan2_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, rhs_type),
                   ie_device, precision, ir_version, freeze_model=False, trace_model=True, fx_kind="aten.arctan2")


class TestAtan2ZeroEdgeCases(PytorchLayerTest):
    """Test atan2 with zero in the inputs - regression test for atan2(0, 0)
    returning NaN.

    The common_translators atan2 decomposition computes atan(y/x), which for
    x=0, y=0 produces atan(NaN) = NaN. Test handling for all IEEE 754
    signed-zero cases:
      atan2(+0, +0) = +0
      atan2(-0, +0) = -0
      atan2(+0, -0) = +π
      atan2(-0, -0) = -π
    """

    def _prepare_input(self):
        if self.zero_case == "both_positive_zero":
            y = torch.zeros(1, dtype=torch.float32).numpy()
            x = torch.zeros(1, dtype=torch.float32).numpy()
        elif self.zero_case == "mixed_zeros_and_axes":
            y = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0], dtype=torch.float32).numpy()
            x = torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0], dtype=torch.float32).numpy()
        elif self.zero_case == "all_zeros":
            y = torch.zeros(3, dtype=torch.float32).numpy()
            x = torch.zeros(3, dtype=torch.float32).numpy()
        elif self.zero_case == "x_zero_various_y":
            y = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32).numpy()
            x = torch.zeros(3, dtype=torch.float32).numpy()
        elif self.zero_case == "signed_neg_zero_x":
            y = torch.tensor([0.0, -0.0], dtype=torch.float32).numpy()
            x = torch.tensor([-0.0, -0.0], dtype=torch.float32).numpy()
        elif self.zero_case == "signed_neg_zero_y":
            y = torch.tensor([-0.0, -0.0], dtype=torch.float32).numpy()
            x = torch.tensor([0.0, -0.0], dtype=torch.float32).numpy()
        elif self.zero_case == "all_four_signed_zero_combos":
            y = torch.tensor([0.0, -0.0, 0.0, -0.0], dtype=torch.float32).numpy()
            x = torch.tensor([0.0, 0.0, -0.0, -0.0], dtype=torch.float32).numpy()
        return (y, x)

    def create_model(self):
        class aten_atan2(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, lhs, rhs):
                out = torch.arctan2(lhs, rhs)
                # Use reciprocal so that signed-zero differences become
                # observable: +0 -> +inf, -0 -> -inf.
                return 1.0 / out

        return aten_atan2(), "aten::atan2"

    @pytest.mark.parametrize("zero_case", [
        "both_positive_zero",
        "mixed_zeros_and_axes",
        "all_zeros",
        "x_zero_various_y",
        "signed_neg_zero_x",
        "signed_neg_zero_y",
        "all_four_signed_zero_combos",
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_atan2_zero_edge_cases(self, ie_device, precision, ir_version, zero_case):
        self.zero_case = zero_case
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   use_convert_model=True, fx_kind="aten.arctan2")
