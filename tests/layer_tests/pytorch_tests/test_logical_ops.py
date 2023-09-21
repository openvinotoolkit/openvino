import numpy as np
import pytest
from pytorch_layer_test_class import PytorchLayerTest

class TestLogicalOp(PytorchLayerTest):

    def _prepare_input(self, out, unary):
        x = np.random.randint(0, 2, (1, 10)).astype(bool)
        if unary:
            return (x, ) if not out else (x, np.zeros_like(x).astype(bool))
        y = np.random.randint(0, 2, (1, 10)).astype(bool)
        if not out:
            return x, y
        return x, y, np.zeros_like(x).astype(bool)

    def create_model(self, op_name, out):
        import torch

        ops = {
            "and": torch.logical_and,
            "or": torch.logical_or,
            "xor": torch.logical_xor,
            "not": torch.logical_not
        }
        op = ops[op_name]
        class aten_logical(torch.nn.Module):

            def __init__(self, op, out) -> None:
                super().__init__()
                self.op = op
                if op == torch.logical_not:
                    self.forward = self.forward_not
                if out:
                    self.forward = self.forward_out if not op == torch.logical_not else self.forward_not_out

            def forward(self, tensor_a, tensor_b):
                return self.op(tensor_a, tensor_b)


            def forward_out(self, tensor_a, tensor_b, out):
                return self.op(tensor_a, tensor_b, out=out), out

            def forward_not(self, tensor_a):
                return self.op(tensor_a)

            def forward_not_out(self, tensor_a, out):
                return self.op(tensor_a, out=out), out

        ref_net = None

        return aten_logical(op, out), ref_net, f"aten::logical_{op_name}"
 

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("op_type", ["and", "or", "not", "xor"])
    @pytest.mark.parametrize("out", [True, False])
    def test_and_tensor(self, op_type, out, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type, out),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "unary": op_type == "not"})