# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest

OPS = {
    "aten::rsqrt": torch.rsqrt,
    "aten::sqrt": torch.sqrt,
    "aten::exp": torch.exp,
    "aten::exp_": torch.exp_,
    "aten::relu": torch.relu,
    "aten::relu_": torch.relu_,
    "aten::ceil": torch.ceil,
    "aten::ceil_": torch.ceil_,
    "aten::floor": torch.floor,
    "aten::floor_": torch.floor_,
    "aten::sigmoid": torch.sigmoid,
    "aten::sigmoid_": torch.sigmoid_,
    "aten::cos": torch.cos,
    "aten::cos_": torch.cos_,
    "aten::sin": torch.sin,
    "aten::sin_": torch.sin_,
    "aten::tan": torch.tan,
    "aten::tan_": torch.tan_,
    "aten::cosh": torch.cosh,
    "aten::cosh_": torch.cosh_,
    "aten::sinh": torch.sinh,
    "aten::sinh_": torch.sinh_,
    "aten::tanh": torch.tanh,
    "aten::tanh_": torch.tanh_,
    "aten::acos": torch.acos,
    "aten::acos_": torch.acos_,
    "aten::asin": torch.asin,
    "aten::asin_": torch.asin_,
    "aten::atan": torch.atan,
    "aten::atan_": torch.atan_,
    "aten::acosh": torch.acosh,
    "aten::acosh_": torch.acosh_,
    "aten::asinh": torch.asinh,
    "aten::asinh_": torch.asinh_,
    "aten::atanh": torch.atanh,
    "aten::atanh_": torch.atanh_
}

class unary_op_net(torch.nn.Module):
    def __init__(self, op, dtype):
        super(unary_op_net, self).__init__()
        self.dtype = dtype
        self.op = op

    def forward(self, x):
        x1 = x.to(self.dtype)
        y = self.op(x1)
        return y, x1


class TestUnaryOp(PytorchLayerTest):
    def _prepare_input(self):
        # random number in range [1, 11)
        x = torch.rand(2, 10) * 10 + 1
        return (x.to(self.dtype).numpy(),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int8, torch.uint8, torch.int32, torch.int64])
    @pytest.mark.parametrize("op_type",
    [
        "aten::rsqrt",
        "aten::sqrt",
        "aten::exp",
        "aten::relu",
        "aten::relu_",
        "aten::ceil",
        "aten::ceil_",
        "aten::floor",
        "aten::floor_",
        "aten::sigmoid",
        # trigonometry
        "aten::cos",
        "aten::sin",
        "aten::tan",
        "aten::cosh",
        "aten::sinh",
        "aten::tanh",
        "aten::acos",
        "aten::asin",
        "aten::atan",
        "aten::acosh",
        "aten::asinh",
        "aten::atanh"
    ])
    def test_unary_op(self, op_type, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(unary_op_net(OPS[op_type], dtype), None, op_type,
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("op_type",
    [
        # some pytorch inplace ops do not support int
        "aten::exp_",
        "aten::sigmoid_",
        # trigonometry
        "aten::cos_",
        "aten::sin_",
        "aten::tan_",
        "aten::cosh_",
        "aten::sinh_",
        "aten::tanh_",
        "aten::acos_",
        "aten::asin_",
        "aten::atan_",
        "aten::acosh_",
        "aten::asinh_",
        "aten::atanh_"
    ])
    def test_unary_op_float(self, op_type, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(unary_op_net(OPS[op_type], dtype), None, op_type,
                   ie_device, precision, ir_version)


class glu_net(torch.nn.Module):
    def __init__(self, dim, dtype):
        super(glu_net, self).__init__()
        self.dtype = dtype
        self.dim = dim

    def forward(self, x):
        y = F.glu(x.to(self.dtype), dim=self.dim)
        return y


class TestGluOp(PytorchLayerTest):
    def _prepare_input(self):
        # random number in range [1, 11)
        x = torch.rand(2, 4, 10, 10) * 10 + 1
        return (x.to(self.dtype).numpy(),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dim", [0, 1, 2, 3, -1, -2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_glu(self, dim, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(glu_net(dim, dtype), None, "aten::glu",
                   ie_device, precision, ir_version)
