# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize(
    "input_shape_rhs",
    [
        [2, 5, 3, 4],
        [1, 5, 3, 4],
        [1]
    ]
)
class TestRemainder(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 5, 3, 4), self.input_rhs)

    def create_model(self):
        class aten_remainder(torch.nn.Module):
            def forward(self, lhs, rhs):
                return torch.remainder(lhs, rhs)


        return aten_remainder(), "aten::remainder"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_remainder(self, ie_device, precision, ir_version, input_shape_rhs):
        self.input_rhs = self.random.randn(*input_shape_rhs)
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)


class TestRemainderTypes(PytorchLayerTest):
    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (self.random.randint(2, 5, size=self.rhs_shape, dtype=self.rhs_type),)
        elif len(self.rhs_shape) == 0:
            return (10 * self.random.randn(*self.lhs_shape, dtype=self.lhs_type),)
        return (
            10 * self.random.randn(*self.lhs_shape, dtype=self.lhs_type),
            self.random.randint(2, 5, size=self.rhs_shape, dtype=self.rhs_type),
        )

    def create_model(self, lhs_type, lhs_shape, rhs_type, rhs_shape):
        class aten_remainder(torch.nn.Module):
            def __init__(self, lhs_type, lhs_shape, rhs_type, rhs_shape):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type
                if len(lhs_shape) == 0:
                    self.forward = self.forward1
                elif len(rhs_shape) == 0:
                    self.forward = self.forward2
                else:
                    self.forward = self.forward3

            def forward1(self, rhs):
                return torch.remainder(torch.tensor(3).to(self.lhs_type), rhs.to(self.rhs_type))

            def forward2(self, lhs):
                return torch.remainder(lhs.to(self.lhs_type), torch.tensor(3).to(self.rhs_type))

            def forward3(self, lhs, rhs):
                return torch.remainder(lhs.to(self.lhs_type), rhs.to(self.rhs_type))


        return aten_remainder(lhs_type, lhs_shape, rhs_type, rhs_shape), "aten::remainder"

    @pytest.mark.parametrize(
        ("lhs_type", "rhs_type"),
        [
            [torch.int32, torch.int64],
            [torch.int32, torch.float32],
            [torch.int32, torch.float64],
            [torch.int64, torch.int32],
            [torch.int64, torch.float32],
            [torch.int64, torch.float64],
            [torch.float32, torch.int32],
            [torch.float32, torch.int64],
            [torch.float32, torch.float64],
        ],
    )
    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape"),
        [
            ([2, 3], [2, 3]),
            ([2, 3], []),
            ([], [2, 3]),
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_remainder_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(
            *self.create_model(lhs_type, lhs_shape, rhs_type, rhs_shape),
            ie_device,
            precision,
            ir_version,
        )
