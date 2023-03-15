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
        return torch.zeros(self.size, 
            dtype = self.dtype
        )

class aten_zeros_complex(torch.nn.Module):
    def __init__(self, size, dtype) -> None:
        torch.nn.Module.__init__(self)
        self.size = size
        self.dtype = dtype

    def forward(self, x):
        return torch.zeros(self.size, 
            dtype = self.dtype,
            layout = torch.strided, # unused
            device = None, # unused
            requires_grad = False # unused
        )

class TestChunk(PytorchLayerTest):
    def _prepare_input(self):
        return (None,)
    
    @pytest.mark.parametrize("size", [
        [100],
        [100,100],
        [9,18],
        [3,3,3]
    ])
    @pytest.mark.parametrize("dtype", [
        torch.int32,
        torch.int16,
        torch.bool,
        torch.float32,
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_zeros_complex(self, size, dtype, ie_device, precision, ir_version):
        self._test(aten_zeros_complex(size, dtype), None, ["aten::zeros"], 
                ie_device, precision, ir_version)

    @pytest.mark.parametrize("size", [
        [100],
        [100,100],
        [9,18],
        [3,3,3]
    ])
    @pytest.mark.parametrize("dtype", [
        torch.int32,
        torch.int16,
        torch.bool,
        torch.float32,
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_zeros(self, size, dtype, ie_device, precision, ir_version):
        self._test(aten_zeros(size, dtype), None, ["aten::zeros"], 
                ie_device, precision, ir_version)
