# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSub(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_data

    def create_model(self, op_type):
        ops = {
            "sub": torch.sub,
            "rsub": torch.rsub
        }

        op = ops[op_type]

        class aten_sub(torch.nn.Module):
            def __init__(self, op):
                self.op = op

            def forward(self, x, y, alpha: float):
                return self.op(x, y, alpha=alpha)

        ref_net = None

        return aten_sub(op), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize('input_data', [(np.random.randn(2, 3, 4).astype(np.float32),
                                             np.random.randn(
                                                 2, 3, 4).astype(np.float32),
                                             np.random.randn(1)),
                                            (np.random.randn(4, 2, 3).astype(np.float32),
                                             np.random.randn(
                                                 1, 2, 3).astype(np.float32),
                                             np.random.randn(1)), ])
    @pytest.mark.parametrize("case", ["sub", "rsub"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sub(self, ie_device, precision, ir_version, input_data, case):
        self.input_data = input_data
        self._test(*self.create_model(case), ie_device, precision, ir_version)


class TestSubTypes(PytorchLayerTest):

    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randn(self.rhs_shape).to(self.rhs_type).numpy(),)
        elif len(self.rhs_shape) == 0:
            return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),)
        return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randn(self.rhs_shape).to(self.rhs_type).numpy())

    def create_model(self, op_type, lhs_type, lhs_shape, rhs_type, rhs_shape):
        ops = {
            "sub": torch.sub,
            "rsub": torch.rsub
        }

        op = ops[op_type]

        class aten_sub(torch.nn.Module):
            def __init__(self, op, lhs_type, lhs_shape, rhs_type, rhs_shape):
                super().__init__()
                self.op = op
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type
                if len(lhs_shape) == 0:
                    self.forward = self.forward1
                elif len(rhs_shape) == 0:
                    self.forward = self.forward2
                else:
                    self.forward = self.forward3

            def forward1(self, rhs):
                return self.op(torch.tensor(3).to(self.lhs_type), rhs.to(self.rhs_type), alpha=2)

            def forward2(self, lhs):
                return self.op(lhs.to(self.lhs_type), torch.tensor(3).to(self.rhs_type), alpha=2)

            def forward3(self, lhs, rhs):
                return self.op(lhs.to(self.lhs_type), rhs.to(self.rhs_type), alpha=2)

        ref_net = None

        return aten_sub(op, lhs_type, lhs_shape, rhs_type, rhs_shape), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.int32, torch.int64],
                              [torch.int32, torch.float32],
                              # [torch.int32, torch.float64], fp64 produce ov error of eltwise constant fold
                              [torch.int64, torch.int32],
                              [torch.int64, torch.float32],
                              # [torch.int64, torch.float64], fp64 produce ov error of eltwise constant fold
                              [torch.float32, torch.int32],
                              [torch.float32, torch.int64],
                              # [torch.float32, torch.float64], fp64 produce ov error of eltwise constant fold
                              ])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], []),
                                                          ([], [2, 3]),
                                                          ])
    @pytest.mark.parametrize("case", ["sub", "rsub"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sub_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape, case):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(case, lhs_type, lhs_shape, rhs_type, rhs_shape),
                   ie_device, precision, ir_version)
