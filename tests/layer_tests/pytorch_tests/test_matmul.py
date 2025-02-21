import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestMatMulOperation(PytorchLayerTest):
    def _prepare_input(self, matrix, vector):
        matrix_input = np.array(matrix, dtype=np.float32)
        vector_input = np.array(vector, dtype=np.float32)
        return matrix_input, vector_input

    def create_model(self, matrix, vector):
        class CustomMatMulOperation(torch.nn.Module):
            def forward(self, matrix, vector):
                return torch.mv(matrix, vector)  

        model_class = CustomMatMulOperation()
        ref_net = None
        return model_class, ref_net, "aten::mv"  

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("matrix, vector, dtype", [
        (np.array([[1, 2], [3, 4]]), np.array([5, 6]), torch.float64),
        (np.array([[0, 0], [0, 0]]), np.array([1, 2]), torch.float32),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([0, 1, 0]), torch.float64),
        (np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([2, 3, 4]), torch.float32),
    ])
    def test_matmul_operation(self, matrix, vector, dtype, ie_device, precision, ir_version):
        matrix_input = torch.tensor(matrix, dtype=torch.float32)
        vector_input = torch.tensor(vector, dtype=torch.float32)

        matrix_input = matrix_input.to(dtype=dtype)
        vector_input = vector_input.to(dtype=dtype)

        self._test(
            *self.create_model(matrix_input, vector_input),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"matrix": matrix_input, "vector": vector_input}
        )
