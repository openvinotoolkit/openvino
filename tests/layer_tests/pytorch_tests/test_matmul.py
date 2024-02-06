import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestMVOperation(PytorchLayerTest):
    def _prepare_input(self, matrix, vector):
        matrix_input = np.array(matrix).astype(np.float32)
        vector_input = np.array(vector).astype(np.float32)
        return [matrix_input, vector_input]

    def create_model(self, matrix, vector):
        class CustomMVOperation(torch.nn.Module):
            def forward(self, matrix, vector):
                return matrix * vector  # Element-wise multiplication as per the provided code

        model_class = CustomMVOperation()
        ref_net = None
        return model_class, ref_net, "aten::mv"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("matrix, vector", [
        (np.array([[1, 2], [3, 4]]), np.array([5, 6])),
        (np.array([[0, 0], [0, 0]]), np.array([1, 2])),
        # Add more test cases as needed
    ])
    def test_mv_operation(self, matrix, vector, ie_device, precision, ir_version):
        self._test(
            *self.create_model(matrix, vector),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"matrix": matrix, "vector": vector}
        )
