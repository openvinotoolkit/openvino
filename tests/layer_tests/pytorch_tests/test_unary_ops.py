# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export

OPS = {
    "aten::abs": torch.abs,
    "aten::abs_": torch.abs_,
    "aten::rsqrt": torch.rsqrt,
    "aten::rsqrt_": torch.rsqrt_,
    "aten::sqrt": torch.sqrt,
    "aten::sqrt_": torch.sqrt_,
    "aten::erf": torch.erf,
    "aten::erf_": torch.erf_,
    "aten::erfc": torch.erfc,
    "aten::erfc_": torch.erfc_,
    "aten::exp": torch.exp,
    "aten::exp_": torch.exp_,
    "aten::expm1": torch.expm1,
    "aten::expm1_": torch.expm1_,
    "aten::relu": torch.relu,
    "aten::relu_": torch.relu_,
    "aten::ceil": torch.ceil,
    "aten::ceil_": torch.ceil_,
    "aten::floor": torch.floor,
    "aten::floor_": torch.floor_,
    "aten::sigmoid": torch.sigmoid,
    "aten::sigmoid_": torch.sigmoid_,
    "aten::reciprocal": torch.reciprocal,
    "aten::reciprocal_": torch.reciprocal_,
    "aten::relu6": F.relu6,
    "aten::selu": F.selu,
    "aten::silu": F.silu,
    "aten::log": torch.log,
    "aten::log_": torch.log_,
    "aten::log2": torch.log2,
    "aten::log2_": torch.log2_,
    "aten::log10": torch.log10,
    "aten::log10_": torch.log10_,
    "aten::log1p": torch.log1p,
    "aten::log1p_": torch.log1p_,
    "aten::log_sigmoid": F.logsigmoid,
    "aten::mish": F.mish,
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
    "aten::atanh_": torch.atanh_,
    "aten::hardswish": F.hardswish
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


class unary_op_out_net(torch.nn.Module):
    def __init__(self, op, dtype):
        super(unary_op_out_net, self).__init__()
        self.dtype = dtype
        self.op = op

    def forward(self, x):
        x1 = x.to(self.dtype)
        y = self.op(x1)
        z = torch.empty_like(y)
        y1 = self.op(x1, out=z)
        return y1, z


class unary_func_op_inplace_net(torch.nn.Module):
    def __init__(self, op, dtype):
        super(unary_func_op_inplace_net, self).__init__()
        self.dtype = dtype
        self.op = op

    def forward(self, x):
        x1 = x.to(self.dtype)
        y = self.op(x1, inplace=True)
        return y, x1


class TestUnaryOp(PytorchLayerTest):
    def _prepare_input(self):
        # random number in range [1, 11)
        x = torch.rand(2, 10) * 10 + 1
        return (x.to(self.dtype).numpy(),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int8, torch.uint8, torch.int32, torch.int64])
    @pytest.mark.parametrize("op_type",
                             [
                                 "aten::abs",
                                 "aten::rsqrt",
                                 "aten::sqrt",
                                 "aten::erf",
                                 "aten::erfc",
                                 "aten::exp",
                                 "aten::expm1",
                                 "aten::relu",
                                 skip_if_export("aten::relu_"),
                                 "aten::ceil",
                                 skip_if_export("aten::ceil_"),
                                 "aten::floor",
                                 skip_if_export("aten::floor_"),
                                 "aten::sigmoid",
                                 "aten::reciprocal",
                                 "aten::log",
                                 "aten::log2",
                                 "aten::log10",
                                 "aten::log1p",
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
        if self.use_torch_export() and op_type == "aten::atanh" and dtype in [torch.int8, torch.int32, torch.int64]:
            pytest.xfail(reason="torch.export after 2.4.0 doesn't support unsigned int types for atanh in some configurations")
        self._test(unary_op_net(OPS[op_type], dtype), None, op_type,
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("op_type",
                             [
                                 # some pytorch inplace ops do not support int
                                 "aten::abs_",
                                 "aten::erf_",
                                 "aten::erfc_",
                                 "aten::exp_",
                                 "aten::rsqrt_",
                                 "aten::sqrt_",
                                 "aten::expm1_",
                                 "aten::sigmoid_",
                                 "aten::reciprocal_",
                                 "aten::relu6",
                                 "aten::selu",
                                 "aten::silu",
                                 "aten::log_sigmoid",
                                 "aten::log_",
                                 "aten::log2_",
                                 "aten::log10_",
                                 "aten::log1p_",
                                 "aten::mish",
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
                                 "aten::atanh_",
                                 "aten::hardswish"
                             ])
    def test_unary_op_float(self, op_type, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(unary_op_net(OPS[op_type], dtype), None, op_type,
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int8, torch.uint8, torch.int32, torch.int64])
    @pytest.mark.parametrize("op_type",
                             [
                                 "aten::abs",
                                 "aten::rsqrt",
                                 "aten::sqrt",
                                 "aten::erf",
                                 "aten::erfc",
                                 "aten::exp",
                                 "aten::expm1",
                                 "aten::relu",
                                 "aten::ceil",
                                 "aten::floor",
                                 "aten::sigmoid",
                                 "aten::reciprocal",
                                 "aten::log",
                                 "aten::log2",
                                 "aten::log10",
                                 "aten::log1p",
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
    def test_unary_op_out(self, op_type, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(unary_op_out_net(OPS[op_type], dtype), None, op_type,
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("op_type",
                             [
                                 "aten::relu6",
                                 "aten::selu",
                                 "aten::silu",
                                 "aten::hardswish",
                                 "aten::mish",
                             ])
    def test_unary_func_op_inplace(self, op_type, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(unary_func_op_inplace_net(OPS[op_type], dtype), None, op_type + "_",
                   ie_device, precision, ir_version)
