import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class aten_relu6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6(
    def forward(self, x):
        return self.relu6(x)

class TestReLU6(PytorchLayerTest):
    def _prepare_input(self):
        data = np.random.uniform(-5.0, 10.0, size=(1, 3, 8, 8)).astype(np.float32)
        return (torch.from_numpy(data),) 
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_relu6(self, ie_device, precision, ir_version):
        model = aten_relu6()
        self._test(model, None, "aten::relu6", ie_device, precision, ir_version)
