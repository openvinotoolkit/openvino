import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class aten_lstm_cell(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)

    def forward(self, x, hx, cx):
        return self.lstm_cell(x, (hx, cx))

class TestLSTMCell(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 10).astype(np.float32), 
                np.random.randn(2, 20).astype(np.float32), 
                np.random.randn(2, 20).astype(np.float32))

    @pytest.mark.parametrize("input_size,hidden_size", [(10, 20)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_lstm_cell(self, input_size, hidden_size, ie_device, precision, ir_version):
        self._test(aten_lstm_cell(input_size, hidden_size), 
                   None, 
                   "aten::lstm_cell", 
                   ie_device, 
                   precision, 
                   ir_version,
                   trace_model=False) 
