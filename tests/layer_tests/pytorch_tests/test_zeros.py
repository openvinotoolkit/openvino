import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_zeros(torch.nn.Module):
    def __init__(self, size, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.size = size
        self.dtype = dtype

    def forward(self, x):
        return torch.zeros([self.size], 
            dtype = self.dtype
        )
    
class TestChunk(PytorchLayerTest):
    def _prepare_input(self):
        return (None,)
    
    @pytest.mark.parametrize("dtype", [
        torch.int32,
        torch.int16,
        torch.bool,
        torch.float32,
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_zeros(self, dtype, ie_device, precision, ir_version):
        self._test(aten_zeros(np.random.randint(100), dtype), None, ["aten::zeros"], 
                ie_device, precision, ir_version)
