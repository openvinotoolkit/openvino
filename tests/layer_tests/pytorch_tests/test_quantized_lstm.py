import numpy as np
import pytest
import torch

from openvino._pyopenvino import OpConversionFailure
from pytorch_layer_test_class import PytorchLayerTest


class TestQuantizedLSTM(PytorchLayerTest):

    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_quantized_lstm_dynamic_not_supported(self, ie_device, precision, ir_version):

        input_size = 8
        hidden_size = 16

        class QuantizedLSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)

            def forward(self, x):
                return self.lstm(x)[0]

        model = QuantizedLSTMModel()

        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.LSTM},
            dtype=torch.qint8
        )

        self.input_tensor = np.random.randn(1, 5, input_size).astype(np.float32)

        with pytest.raises(OpConversionFailure, match="Quantized LSTM is not supported"):
            self._test(
                quantized_model,
                None,
                "aten::quantized_lstm",
                ie_device,
                precision,
                ir_version,
                trace_model=True
            )
