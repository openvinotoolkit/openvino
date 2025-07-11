import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class aten_quantized_relu6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.zero_point = 0
        self.relu6 = torch.nn.quantized.ReLU6()

    def forward(self, x):
        x_q = torch.quantize_per_tensor(x, scale=self.scale, zero_point=self.zero_point, dtype=torch.quint8)
        out_q = self.relu6(x_q)
        return out_q.dequantize()

class TestQuantizedReLU6(PytorchLayerTest):
    def _prepare_input(self):
        data = np.random.uniform(-5.0, 10.0, size=(1, 3, 8, 8)).astype(np.float32)
        return (torch.from_numpy(data),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_quantized_relu6(self, ie_device, precision, ir_version):
        model = aten_quantized_relu6()
        self._test(model, None, "quantized::relu6", ie_device, precision, ir_version)
