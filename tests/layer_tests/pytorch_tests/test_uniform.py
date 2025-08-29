import pytest
import torch
import numpy as np
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

    @pytest.mark.parametrize("input_shape", [(2, 3), (4, 5)])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("from_val,to_val", [(0.0, 1.0), (-1.0, 1.0)])
    def test_uniform(
        self, input_shape, dtype, from_val, to_val, ie_device, precision, ir_version
    ):
        self._test(
            *self.create_model(from_val, to_val),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype},
        )
