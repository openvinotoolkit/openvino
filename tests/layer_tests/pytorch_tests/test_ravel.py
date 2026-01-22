
import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestRavel(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 3, 4).astype(np.float32),)

    class RavelModel(torch.nn.Module):
        def forward(self, x):
            return torch.ravel(x)

    def test_ravel(self, ie_device, precision, ir_version):
        model = self.RavelModel()
        self._test(model, None, "aten::ravel", 
                   ie_device, precision, ir_version)
