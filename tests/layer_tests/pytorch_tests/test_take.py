
import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestTake(PytorchLayerTest):
    def _prepare_input(self):
        return (torch.randn(4, 5).numpy(), np.array([0, 19, 5, 2], dtype=np.int64))

    def create_model(self):
        class idx_model(torch.nn.Module):
            def forward(self, x, idx):
                return torch.take(x, idx)

        ref_net = None
        return idx_model(), ref_net, "aten::take"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_take(self, ie_device, precision, ir_version):
        self._test(self.create_model(), None, ie_device, precision, ir_version)
