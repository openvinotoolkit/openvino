import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestTensordot(PytorchLayerTest):
    def _prepare_input(self, shape_a=(2, 3), shape_b=(3, 2)):
        return (
            np.random.randn(*shape_a).astype(np.float32),
            np.random.randn(*shape_b).astype(np.float32),
            1  # dimensione da sommare
        )

    def create_model(self):
        class TensordotModel(torch.nn.Module):
            def forward(self, a, b, dims):
                return torch.tensordot(a, b, dims=dims)
        return TensordotModel(), None, "aten::tensordot"

    @pytest.mark.parametrize("shape_a, shape_b", [
        ((2, 3), (3, 2)),
        ((4, 5), (5, 4)),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tensordot(self, shape_a, shape_b, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"shape_a": shape_a, "shape_b": shape_b})
