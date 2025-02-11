import pytest

from pytorch_layer_test_class import PytorchLayerTest

class TestTakeAlongDim(PytorchLayerTest):
    def _prepare_input(self, m, n, max_val, out=False, flattenize=False):
        import numpy as np
        shape = (m, n) if not flattenize else (m * n,)
        index = np.random.randint(0, max_val, shape).astype(np.int64)
        inp = np.random.randn(m, n).astype(np.float32)
        if out:
            axis = int(max_val == n)
            if flattenize:
                out = np.zeros_like(np.take(inp, index))
            else:
                out = np.zeros_like(np.take(inp, index, axis))
            return (inp, index, out)
        return (inp, index)

    def create_model(self, axis, out):
        import torch

        class aten_take_along_dim(torch.nn.Module):
            def __init__(self, axis, out=False):
                super(aten_take_along_dim, self).__init__()
                self.axis = axis
                if self.axis is None:
                    self.forward = self.forward_no_dim
                if out:
                    self.forward = self.forward_out if self.axis is not None else self.forward_no_dim_out

            def forward(self, x, index):
                return torch.take_along_dim(x, index, dim=self.axis)

            def forward_out(self, x, index, out):
                return torch.take_along_dim(x, index, dim=self.axis, out=out), out
    
            def forward_no_dim(self, x, index):
                return torch.take_along_dim(x, index)

            def forward_no_dim_out(self, x, index, out):
                return torch.take_along_dim(x, index, out=out)

        return aten_take_along_dim(axis, out), None, "aten::take_along_dim"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("m", [2, 10, 100])
    @pytest.mark.parametrize("n", [2, 10, 100])
    @pytest.mark.parametrize("axis", [0, 1, None])
    @pytest.mark.parametrize("out", [True, False])
    def test_gather(self, m, n, axis, out, ie_device, precision, ir_version):
        self._test(*self.create_model(axis, out), ie_device, precision, ir_version, kwargs_to_prepare_input={
            "m": m, "n": n, "max_val": m if axis == 0 else n, "out": out, "flattenize": axis is None
        })
