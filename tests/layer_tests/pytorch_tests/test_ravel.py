import pytest
import torch
import numpy as np
from common.layer_test_class import CommonLayerTest
from pytorch_layer_test_class import PytorchLayerTest

# Define the test class
class TestRavel(PytorchLayerTest):
    def _prepare_input(self):
        return (torch.randn(2, 3, 4),)

    def create_model(self):
        class RavelModel(torch.nn.Module):
            def forward(self, x):
                return torch.ravel(x)

        return RavelModel(), None

    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.parametrize("device", ["CPU", "GPU"])
    def test_ravel(self, device):
        self._test(*self.create_model(), ie_device=device)
