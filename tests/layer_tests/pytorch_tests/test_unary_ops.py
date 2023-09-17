# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


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
    @pytest.mark.parametrize("op,op_type", [
        (torch.rsqrt, "aten::rsqrt"),
        (torch.sqrt, "aten::sqrt"),
        (torch.exp, "aten::exp"),
        (torch.relu, "aten::relu"),
        (torch.relu_, "aten::relu_"),
        (torch.ceil, "aten::ceil"),
        (torch.ceil_, "aten::ceil_"),
        (torch.floor, "aten::floor"),
        (torch.floor_, "aten::floor_"),
        (torch.sigmoid, "aten::sigmoid"),
        # trigonometry
        (torch.cos, "aten::cos"),
        (torch.sin, "aten::sin"),
        (torch.tan, "aten::tan"),
        (torch.cosh, "aten::cosh"),
        (torch.sinh, "aten::sinh"),
        (torch.tanh, "aten::tanh"),
        (torch.acos, "aten::acos"),
        (torch.asin, "aten::asin"),
        (torch.atan, "aten::atan"),
        (torch.acosh, "aten::acosh"),
        (torch.asinh, "aten::asinh"),
        (torch.atanh, "aten::atanh"),
    ])
    def test_unary_op(self, op, op_type, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(unary_op_net(op, dtype), None, op_type,
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("op,op_type", [
        # some pytorch inplace ops do not support int
        (torch.exp_, "aten::exp_"),
        (torch.sigmoid_, "aten::sigmoid_"),
        # trigonometry
        (torch.cos_, "aten::cos_"),
        (torch.sin_, "aten::sin_"),
        (torch.tan_, "aten::tan_"),
        (torch.cosh_, "aten::cosh_"),
        (torch.sinh_, "aten::sinh_"),
        (torch.tanh_, "aten::tanh_"),
        (torch.acos_, "aten::acos_"),
        (torch.asin_, "aten::asin_"),
        (torch.atan_, "aten::atan_"),
        (torch.acosh_, "aten::acosh_"),
        (torch.asinh_, "aten::asinh_"),
        (torch.atanh_, "aten::atanh_"),
    ])
    def test_unary_op_float(self, op, op_type, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(unary_op_net(op, dtype), None, op_type,
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
