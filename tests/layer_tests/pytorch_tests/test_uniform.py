import pytest
import torch
import numpy as np
import numpy.testing as npt
import openvino as ov
from pytorch_layer_test_class import PytorchLayerTest


class TestUniform(PytorchLayerTest):
    def _prepare_input(self, input_shape=(2, 2), dtype=np.float32):
        return (np.random.randn(*input_shape).astype(dtype),)

    def create_model(self, from_val, to_val):
        class aten_uniform(torch.nn.Module):
            def __init__(self, from_val, to_val):
                super().__init__()
                self.from_val = from_val
                self.to_val = to_val

            def forward(self, x):
                return x.uniform_(self.from_val, self.to_val)

        ref_net = None
        return aten_uniform(from_val, to_val), ref_net, "aten::uniform_"

    @pytest.mark.parametrize("input_shape", [(1000, 100), (1000, 100)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("from_val,to_val", [(0.0, 1.0), (-1.0, 1.0)])
    
    def test_uniform(self, input_shape, dtype, from_val, to_val, ie_device, precision, ir_version):
        fw_model, ref_net, op_type = self.create_model(from_val, to_val)
        inputs = self._prepare_input(input_shape=input_shape, dtype=dtype)
        example_input = tuple(torch.from_numpy(inp) for inp in inputs)

        ov_model = ov.convert_model(
            input_model=fw_model,
            example_input=example_input,
            input=[inp.shape for inp in example_input],
        )

        if ie_device == 'GPU' and precision == 'FP32':
            config = {'INFERENCE_PRECISION_HINT': 'f32'}
        else:
            config = {}
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)

        fw_output = model(*example_input).detach().numpy()
        ov_output = compiled_model(inputs)[0]

        x_min, x_max = from_val, to_val
        hist_fw, _ = np.histogram(fw_output, bins=100, range=(x_min, x_max))
        hist_ov, _ = np.histogram(ov_output, bins=100, range=(x_min, x_max))

        npt.assert_allclose(hist_fw, hist_ov, atol=0.2 * hist_fw.max(), rtol=0.2)
