
import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestTake(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(4, 3).astype(np.float32), np.array([0, 11, 2, 5], dtype=np.int64))

    class TakeModel(torch.nn.Module):
        def forward(self, x, idx):
            return torch.take(x, idx)

    def test_take(self, ie_device, precision, ir_version):
        model = self.TakeModel()
        self._test(model, None, "aten::take", 
                   ie_device, precision, ir_version)
