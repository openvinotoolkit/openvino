import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest, flattenize_inputs

class TestRandperm(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([self.n], dtype=np.int64),)

    def create_model(self, n, num_inputs, dtype_value=None):
        class aten_randperm(torch.nn.Module):
            def __init__(self, n, num_inputs, dtype_value):
                super().__init__()
                self.n = torch.tensor(n, dtype=torch.int64)
                self.num_inputs = num_inputs
                self.dtype = torch.int64 if dtype_value == 4 else None
            
            def forward(self, x):
                if self.num_inputs == 1:
                    return torch.randperm(self.n)
                elif self.num_inputs == 2:
                    return torch.randperm(self.n, dtype=self.dtype)
                elif self.num_inputs == 5:
                    return torch.randperm(self.n, dtype=self.dtype, layout=torch.strided, 
                                          device=x.device, pin_memory=False)
                raise ValueError("Invalid num_inputs")
        
        return aten_randperm(n, num_inputs, dtype_value), None, "aten::randperm"

    @pytest.mark.parametrize(("n", "num_inputs", "dtype_value"), [
        (0, 1, None),
        (1, 1, None),
        (5, 1, None),
        (5, 2, 4),
        (5, 5, 4),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_randperm(self, n, num_inputs, dtype_value, ie_device, precision, ir_version):
        self.n = n
        model, ref_net, op = self.create_model(n, num_inputs, dtype_value)
        inputs = self._prepare_input()
        torch_inputs = [torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in inputs]
        ov_inputs = flattenize_inputs(inputs)
        smodel, converted_model = self.convert_directly_via_frontend(
            model, torch_inputs, trace_model=True, dynamic_shapes=False, ov_inputs=ov_inputs, freeze_model=True
        )
        from openvino import Core
        core = Core()
        compiled_model = core.compile_model(converted_model, ie_device)
        
        ov_output = compiled_model(ov_inputs)[0]
        if n > 0:
            assert ov_output.shape[0] == n, f"Output shape {ov_output.shape} does not match expected ({n},)"
            assert np.array_equal(np.sort(ov_output), np.arange(n)), f"Output is not a valid permutation of [0, ..., {n-1}]"
        else:
            assert ov_output.shape[0] == 0, f"Output shape for n=0 should be (0,), got {ov_output.shape}"
