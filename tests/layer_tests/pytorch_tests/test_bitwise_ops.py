# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from packaging import version

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestBitwiseOp(PytorchLayerTest):
    def _prepare_input(self, out, unary, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape):
        choices = np.array([0, 1, 255, 7])
        x = np.random.choice(choices, lhs_shape).astype(lhs_dtype)
        if unary:
            return (x,) if not out else (x, np.zeros_like(x).astype(lhs_dtype))
        y = np.random.choice(choices, rhs_shape).astype(rhs_dtype)
        if not out:
            return x, y
        return x, y, np.zeros_like(x).astype(lhs_dtype) + np.zeros_like(y).astype(rhs_dtype)

    def create_model(self, op_name, out):
        ops = {
            "and": torch.bitwise_and,
            "or": torch.bitwise_or,
            "xor": torch.bitwise_xor,
            "not": torch.bitwise_not,
        }
        op = ops[op_name]

        class aten_bitwise(torch.nn.Module):
            def __init__(self, op, out) -> None:
                super().__init__()
                self.op = op
                if op == torch.bitwise_not:
                    self.forward = self.forward_not
                if out:
                    self.forward = self.forward_out if not op == torch.bitwise_not else self.forward_not_out

            def forward(self, tensor_a, tensor_b):
                return self.op(tensor_a, tensor_b)

            def forward_out(self, tensor_a, tensor_b, out):
                return self.op(tensor_a, tensor_b, out=out), out

            def forward_not(self, tensor_a):
                return self.op(tensor_a)

            def forward_not_out(self, tensor_a, out):
                return self.op(tensor_a, out=out), out

        ref_net = None

        return aten_bitwise(op, out), ref_net, f"aten::bitwise_{op_name}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("op_type", ["and", "or", "not", "xor"])
    @pytest.mark.parametrize("lhs_dtype", ["bool", "int32", "uint8", "int64"])
    @pytest.mark.parametrize("rhs_dtype", ["bool", "int32", "uint8", "int64"])
    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape"),
        [
            ([2, 3], [2, 3]),
            ([2, 3], []),
            ([], [2, 3]),
        ],
    )
    @pytest.mark.parametrize("out", [False, skip_if_export(True)])
    def test_bitwise_mixed_dtypes(
            self, op_type, out, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, ie_device, precision, ir_version
    ):
        if ie_device == "GPU" and (lhs_dtype != "bool" or rhs_dtype != "bool"):
            pytest.xfail(reason="bitwise ops are not supported on GPU")
        self._test(
            *self.create_model(op_type, out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "out": out,
                "unary": op_type == "not",
                "lhs_dtype": lhs_dtype,
                "rhs_dtype": rhs_dtype,
                "lhs_shape": lhs_shape,
                "rhs_shape": rhs_shape,
            },
            freeze_model=False,
            trace_model=True,
        )


class TestBitwiseOperators(PytorchLayerTest):
    def _prepare_input(self, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape):
        choices = np.array([0, 1, 255, 7])
        x = np.random.choice(choices, lhs_shape).astype(lhs_dtype)
        y = np.random.choice(choices, rhs_shape).astype(rhs_dtype)
        return x, y

    def create_model(self):
        class aten_bitwise(torch.nn.Module):
            def forward(self, lhs, rhs):
                return lhs & rhs, ~lhs, lhs | rhs, lhs ^ rhs

        ref_net = None

        return aten_bitwise(), ref_net, ("aten::__and__", "aten::bitwise_not", "aten::__or__", "aten::__xor__")

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("lhs_dtype", ["bool", "int32"])
    @pytest.mark.parametrize("rhs_dtype", ["bool", "int32"])
    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape"),
        [
            ([2, 3], [2, 3]),
            ([2, 3], []),
            ([], [2, 3]),
        ],
    )
    def test_bitwise_operators(self, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, ie_device, precision, ir_version):
        if ie_device == "GPU" and (lhs_dtype != "bool" or rhs_dtype != "bool"):
            pytest.xfail(reason="bitwise ops are not supported on GPU")
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "lhs_dtype": lhs_dtype,
                "rhs_dtype": rhs_dtype,
                "lhs_shape": lhs_shape,
                "rhs_shape": rhs_shape,
            },
            trace_model=True,
            freeze_model=False,
        )


class TestBitwiseInplaceOp(PytorchLayerTest):
    def _prepare_input(self, lhs_shape, rhs_shape, dtype):
        choices = np.array([0, 1, 255, 7])
        x = np.random.choice(choices, lhs_shape).astype(dtype)
        y = np.random.choice(choices, rhs_shape).astype(dtype)
        return x, y

    def create_model(self, op):
        class aten_bitwise(torch.nn.Module):
            def __init__(self, op) -> None:
                super().__init__()
                if op == "aten::__ior__":
                    self.forward = self.forward_or
                if op == "aten::__iand__":
                    self.forward = self.forward_and
                if op == "aten::__ixor__":
                    self.forward = self.forward_xor

            def forward_or(self, lhs, rhs):
                return lhs.__ior__(rhs)
    
            def forward_and(self, lhs, rhs):
                return lhs.__iand__(rhs)
    
            def forward_xor(self, lhs, rhs):
                return lhs.__ixor__(rhs)

        return aten_bitwise(op), None, op

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", ["bool", "int32"])
    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape"),
        [
            ([2, 3], [2, 3]),
            ([2, 3], []),
        ],
    )
    @pytest.mark.parametrize("op", ["aten::__ior__", "aten::__iand__", "aten::__ixor__"])
    def test_bitwise_operators(self, op, dtype, lhs_shape, rhs_shape, ie_device, precision, ir_version):
        if ie_device == "GPU" and dtype != "bool":
            pytest.xfail(reason="bitwise ops are not supported on GPU")
        self._test(
            *self.create_model(op),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "dtype": dtype,
                "lhs_shape": lhs_shape,
                "rhs_shape": rhs_shape,
            },
            trace_model=True,
            freeze_model=False,
        )