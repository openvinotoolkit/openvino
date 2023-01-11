# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestComp(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 24, 24).astype(np.float32), np.random.randn(1, 3, 24, 24).astype(np.float32))

    def create_model(self, op_type):

        import torch

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
    def test_comp(self, op, ie_device, precision, ir_version):
        self._test(*self.create_model(op), ie_device, precision, ir_version)