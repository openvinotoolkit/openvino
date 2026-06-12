import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestPoisson(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(1.0, 10.0, (2, 3)).astype(np.float32),)

    def create_model(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.poisson(x)
        return Model()

    @pytest.mark.parametrize("ie_device", ["CPU"])
    @pytest.mark.parametrize("precision", ["FP32"])
    def test_poisson(self, ie_device, precision, ir_version):
        self._test(self.create_model(), None, "poisson", ie_device, precision, ir_version, custom_eps=1e10)