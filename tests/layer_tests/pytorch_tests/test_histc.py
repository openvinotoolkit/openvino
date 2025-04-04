import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class aten_histc(torch.nn.Module):
    def __init__(self, bins, min=0, max=0):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.histc(x, bins=self.bins, min=self.min, max=self.max)

class aten_histc_out(torch.nn.Module):
    def __init__(self, bins, min=0, max=0):
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max

    def forward(self, x, out):
        return torch.histc(x, bins=self.bins, min=self.min, max=self.max, out=out), out

class TestHistc(PytorchLayerTest):
    def _prepare_input(self, out=False, x=None):
        if x is None:
            x = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        if not out:
            return (x,)
        out = np.zeros((self.bins,), dtype=np.float32)
        return (x, out)

    @pytest.mark.parametrize("bins,min_val,max_val", [
        (4, 0, 3),
        (2, 1, 2),
        (3, -1, 4)
    ])
    @pytest.mark.parametrize("out", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc(self, bins, min_val, max_val, out, ie_device, precision, ir_version):
        self.bins = bins
        model = aten_histc(bins, min_val, max_val) if not out else aten_histc_out(bins, min_val, max_val)
        example_input = np.array([1.0, 2.0, 1.0], dtype=np.float32)
        self._test(model, None, "aten::histc", ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"out": out, "x": example_input})

    @pytest.mark.parametrize("x_data,bins,min_val,max_val,expected", [
        ([1.5, 2.5, 3.5], 3, 2, 3, [0, 1, 1]),
        ([float('nan'), 2.0, 3.0], 2, 1, 4, [0, 2]),
        ([1.0, 2.0, 3.0, 4.0], 4, 0, 0, [1, 1, 1, 1])
    ])
    @pytest.mark.nightly
    def test_histc_edge_cases(self, ie_device, precision, ir_version, x_data, bins, min_val, max_val, expected):
        self.bins = bins
        model = aten_histc(bins, min_val, max_val)
        x = np.array(x_data, dtype=np.float32)
        self._test(model, None, "aten::histc", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"x": x}, custom_eps=1e-5)