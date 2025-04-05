import pytest
import torch
import numpy as np
import openvino as ov

# 1. Define a simple model that uses torch.flip (reverse along a dimension)
class ReverseModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.flip(x, [self.dim])


# 2. Test signature must include ie_device and precision fixtures
@pytest.mark.parametrize("dim", [0, 1, 2])
def test_reverse_tensor(dim, ie_device, precision):
    # 0) Build model and input, handle precision
    model = ReverseModel(dim).eval()
    x = torch.randn(2, 3, 4)
    if precision == "FP16":
        model = model.half()
        x = x.half()

    # 1) Reference output from PyTorch
    expected = model(x).cpu().numpy()

    # 2) Convert with the PyTorch frontend (will internally trace using example_input)
    ov_model = ov.convert_model(model, example_input=x)
    core = ov.Core()

    # 3) Skip GPU if plugin not available
    if ie_device not in core.available_devices:
        pytest.skip(f"{ie_device} plugin not available")

    # 4) Compile & infer on the requested device
    compiled = core.compile_model(ov_model, device_name=ie_device)

    # 5) Run inference
    result = compiled({compiled.inputs[0]: x.cpu().numpy()})
    output = result[compiled.outputs[0]]

    # 6) Compare with relaxed tolerances for numerical drift
    if precision == "FP32":
        atol, rtol = 1e-5, 1e-3
    else:
        atol, rtol = 1e-3, 1e-2
    assert np.allclose(output, expected, atol=atol, rtol=rtol)





