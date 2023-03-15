import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_zeros(torch.nn.Module):
    def __init__(self, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.dtype = dtype

    def forward(self, input_tensor):
        return torch.zeros(input_tensor, 
            dtype = self.dtype,
            layout = torch.strided,
            device = None,
            requires_grad = False
        )
    
class TestChunk(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randint(100),)
    
    @pytest.mark.parametrize("dtype", [
        torch.int32,
        torch.int16,
        torch.uint16,
        torch.bool,
        torch.float32,
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_zeros(self, dtype, ie_device, precision, ir_version):
        self._test(aten_zeros(dim, chunks), None, ["aten::zeros"], 
                ie_device, precision, ir_version)