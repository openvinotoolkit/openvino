import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class TestRenorm(PytorchLayerTest):
    def _prepare_input(self, m, n):
        return (np.random.randn(m, n).astype(np.float32),)

    def create_model(self, p, dim, maxnorm):
        class aten_renorm(torch.nn.Module):
            def __init__(self, p, dim, maxnorm):
                super(aten_renorm, self).__init__()
                self.p = p
                self.dim = dim
                self.maxnorm = maxnorm

            def forward(self, x):
                return torch.renorm(x, self.p, self.dim, self.maxnorm)

        ref_net = None
        return aten_renorm(p, dim, maxnorm), ref_net, "aten::renorm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("m", [2, 10])
    @pytest.mark.parametrize("n", [3, 5])
    @pytest.mark.parametrize("p", [1, 2, 3])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("maxnorm", [0.5, 1.0, 2.0])
    def test_renorm(self, m, n, p, dim, maxnorm, ie_device, precision, ir_version):
        self._test(*self.create_model(p, dim, maxnorm),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"m": m, "n": n})
        