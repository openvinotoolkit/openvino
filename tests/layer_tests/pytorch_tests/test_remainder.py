# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
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
        return (np.random.randn(2, 5, 3, 4).astype(np.float32), self.input_rhs)

    def create_model(self):
        class aten_remainder(torch.nn.Module):
            def forward(self, lhs, rhs):
                return torch.remainder(lhs, rhs)

        ref_net = None

        return aten_remainder(), ref_net, "aten::remainder"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_remainder(self, ie_device, precision, ir_version, input_shape_rhs):
        self.input_rhs = np.random.randn(*input_shape_rhs).astype(np.float32)
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)


class TestRemainderTypes(PytorchLayerTest):
    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randint(2, 5, self.rhs_shape).to(self.rhs_type).numpy(),)
        elif len(self.rhs_shape) == 0:
            return (10 * torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),)
        return (
            10 * torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),
            torch.randint(2, 5, self.rhs_shape).to(self.rhs_type).numpy(),
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

        ref_net = None

        return aten_remainder(lhs_type, lhs_shape, rhs_type, rhs_shape), ref_net, "aten::remainder"

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
