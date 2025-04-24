# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestComp(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 24, 24).astype(np.float32), np.random.randn(1, 3, 24, 24).astype(np.float32))

    def create_model(self, op_type):
        class aten_eq(torch.nn.Module):
            def forward(self, x, y):
                return x == y

        class aten_ne(torch.nn.Module):
            def forward(self, x, y):
                return x != y

        class aten_lt(torch.nn.Module):
            def forward(self, x, y):
                return x < y

        class aten_gt(torch.nn.Module):
            def forward(self, x, y):
                return x > y

        class aten_le(torch.nn.Module):
            def forward(self, x, y):
                return x <= y

        class aten_ge(torch.nn.Module):
            def forward(self, x, y):
                return x >= y

        ops = {
            "eq": aten_eq,
            "ne": aten_ne,
            "lt": aten_lt,
            "gt": aten_gt,
            "ge": aten_ge,
            "le": aten_le
        }
        model_cls = ops[op_type]

        ref_net = None

        return model_cls(), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "gt", "le", "ge"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_comp(self, op, ie_device, precision, ir_version):
        self._test(*self.create_model(op), ie_device, precision, ir_version, use_convert_model=True)


class TestCompMixedTypes(PytorchLayerTest):

    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randint(0, 3, self.rhs_shape).to(self.rhs_type).numpy(),)
        elif len(self.rhs_shape) == 0:
            return (torch.randint(0, 3, self.lhs_shape).to(self.lhs_type).numpy(),)
        return (torch.randint(0, 3, self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randint(0, 3, self.rhs_shape).to(self.rhs_type).numpy())

    def create_model(self, lhs_type, lhs_shape, rhs_type, rhs_shape, op):

        ops = {
            "eq": torch.eq,
            "ne": torch.ne,
            "lt": torch.lt,
            "gt": torch.gt,
            "ge": torch.ge,
            "le": torch.le
        }

        op_fn = ops[op]

        class aten_comp(torch.nn.Module):
            def __init__(self, lhs_type, lhs_shape, rhs_type, rhs_shape, op_fn):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type
                self.op_fn = op_fn
                if len(lhs_shape) == 0:
                    self.forward = self.forward1
                elif len(rhs_shape) == 0:
                    self.forward = self.forward2
                else:
                    self.forward = self.forward3

            def forward1(self, rhs):
                return self.op_fn(torch.tensor(3).to(self.lhs_type), rhs.to(self.rhs_type))

            def forward2(self, lhs):
                return self.op_fn(lhs.to(self.lhs_type), torch.tensor(3).to(self.rhs_type))

            def forward3(self, lhs, rhs):
                return self.op_fn(lhs.to(self.lhs_type), rhs.to(self.rhs_type))

        ref_net = None

        return aten_comp(lhs_type, lhs_shape, rhs_type, rhs_shape, op_fn), ref_net, f"aten::{op}"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.int32, torch.int64],
                              [torch.int32, torch.float32],
                              [torch.int32, torch.float64],
                              [torch.int64, torch.int32],
                              [torch.int64, torch.float32],
                              [torch.int64, torch.float64],
                              [torch.float32, torch.int32],
                              [torch.float32, torch.int64],
                              [torch.float32, torch.float64],
                              ])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], []),
                                                          ([], [2, 3]),
                                                          ])
    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "gt", "le", "ge"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_eq_mixed_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape, op):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, lhs_shape, rhs_type, rhs_shape, op),
                   ie_device, precision, ir_version)
