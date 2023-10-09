import numpy as np
import pytest
from pytorch_layer_test_class import PytorchLayerTest
import torch


class TestBitwiseOp(PytorchLayerTest):
    def _prepare_input(self, out, unary, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape):
        x = np.random.randint(0, 25, lhs_shape).astype(lhs_dtype)
        if unary:
            return (x,) if not out else (x, np.zeros_like([5, 5]).astype(lhs_dtype))
        y = np.random.randint(0, 25, rhs_shape).astype(rhs_dtype)
        if not out:
            return x, y
        return x, y, np.zeros_like([5, 5]).astype(lhs_dtype)

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
    @pytest.mark.parametrize("out", [False])
    # Tracing required for proper mixed type aligment fails with cases with out param (inplace) - separate it into 2 test cases
    def test_bitwise_mixed_dtypes(
        self, op_type, out, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, ie_device, precision, ir_version
    ):
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

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("op_type", ["and", "or", "not", "xor"])
    @pytest.mark.parametrize(("dtype"), ["bool", "int32", "uint8", "int64"])
    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape"),
        [
            ([2, 3], [2, 3]),
            ([2, 3], []),
            ([], [2, 3]),
        ],
    )
    @pytest.mark.parametrize("out", [True])
    # Tracing required for proper mixed type aligment fails with cases with out param (inplace) - separate it into 2 test cases
    def test_bitwise_out(self, op_type, out, dtype, lhs_shape, rhs_shape, ie_device, precision, ir_version):
        self._test(
            *self.create_model(op_type, out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "out": out,
                "unary": op_type == "not",
                "lhs_dtype": dtype,
                "rhs_dtype": dtype,
                "lhs_shape": lhs_shape,
                "rhs_shape": rhs_shape,
            },
        )
