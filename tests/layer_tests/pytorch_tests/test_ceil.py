import pytest
from pytorch_layer_test_class import PytorchLayerTest

class TestCeil(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self):
        import torch

        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.ceil(x)

        return Model(), "aten::ceil"

    @pytest.mark.precommit
    def test_ceil(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)