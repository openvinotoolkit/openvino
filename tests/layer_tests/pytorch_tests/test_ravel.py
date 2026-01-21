
import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class TestRavel(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 3, 4).astype(np.float32),)

    def create_model(self):
        class RavelModel(torch.nn.Module):
            def forward(self, x):
                return torch.ravel(x)

        return RavelModel(), None, "aten::ravel"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ravel(self, ie_device, precision, ir_version):
        self._test(self.create_model(), None, "aten::ravel", ie_device, precision, ir_version)
