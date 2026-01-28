import pytest
import numpy as np
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestTensordot(PytorchLayerTest):
    def _prepare_input(self, shape_a, shape_b):
        return (
            np.random.randn(*shape_a).astype(np.float32),
            np.random.randn(*shape_b).astype(np.float32)
        )

    def create_model(self, dims):
        class TensordotModel(torch.nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, a, b):
                return torch.tensordot(a, b, dims=self.dims)
        
        return TensordotModel(dims), None, "aten::tensordot"

    @pytest.mark.parametrize("shape_a, shape_b, dims", [
        ((2, 3), (3, 2), 1),
        ((4, 5), (5, 4), 1),
        ((2, 3, 4), (4, 3, 2), 2),
        ((2, 3, 4), (3, 4, 2), ([1, 2], [0, 1])), # Tuple of lists case
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tensordot(self, shape_a, shape_b, dims, ie_device, precision, ir_version):
        self._test(*self.create_model(dims), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"shape_a": shape_a, "shape_b": shape_b})