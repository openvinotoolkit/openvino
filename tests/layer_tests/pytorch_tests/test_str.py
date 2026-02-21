
import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestStr(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 3).astype(np.float32),)

    class StrModel(torch.nn.Module):
        def forward(self, x):
            # Try to prevent folding with conditional
            if x.size(0) > 0:
                s = str(5)
            else:
                s = str(10)
            return torch.tensor(len(s), dtype=torch.float32)

    @pytest.mark.parametrize("model_class", [StrModel])
    def test_str(self, model_class, ie_device, precision, ir_version):
        model = model_class()
        # verify execution
        self._test(model, None, "aten::str", 
                   ie_device, precision, ir_version)
