# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestTrigonom(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 2, 3, 4).astype(np.float32), )

    def create_model(self, op_type):

        import torch
        ops={
            "cos": torch.cos,
            "cos_": torch.cos_,
            "sin": torch.sin,
            "sin_": torch.sin_,
            "tan": torch.tan,
            "tan_": torch.tan_,
            "cosh": torch.cosh,
            "cosh_": torch.cosh_,
            "sinh": torch.sinh,
            "sinh_": torch.sinh_,
            "tanh": torch.tanh,
            "tanh_": torch.tanh_,
            "acos": torch.acos,
            "acos_": torch.acos_,
            "asin": torch.asin,
            "asin_": torch.asin_,
            "atan": torch.atan,
            "atan_": torch.atan_,
            "acosh": torch.acosh,
            "acosh_": torch.acosh_,
            "asinh": torch.asinh,
            "asinh_": torch.asinh_,
            "atanh": torch.atanh,
            "atanh_": torch.atanh_,
        }

        class aten_op(torch.nn.Module):
            def __init__(self, op):
                super(aten_op, self).__init__()
                self.op = op

            def forward(self, x):
                return self.op(x)
        ref_net = None

        return aten_op(ops[op_type]), ref_net, f'aten::{op_type}'

    @ pytest.mark.parametrize("op", [
        "acos", "acos_", "acosh", "acosh_", 
        "asin", "asin_", "asinh", "asinh_", 
        "atan", "atan_", "atanh", "atanh_", 
        "cos", "cos_", "cosh", "cosh_",
        "sin", "sin_", "sinh", "sinh_",
        "tan", "tan_", "tanh", "tanh_"])
    @ pytest.mark.nightly
    def test_mm(self, op, ie_device, precision, ir_version):
        self._test(*self.create_model(op), ie_device, precision, ir_version)
