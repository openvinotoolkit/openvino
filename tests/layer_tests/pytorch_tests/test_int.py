import pytest
from pytorch_layer_test_class import PytorchLayerTest

class TestInt(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.array([1.5], dtype=np.float32),)

    def create_model(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return int(x)

        return Model(), "prim::int"

    @pytest.mark.precommit
    def test_int(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)