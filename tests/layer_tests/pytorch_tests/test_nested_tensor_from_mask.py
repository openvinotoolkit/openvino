import numpy as np
import pytest
import torch
import openvino as ov


def convert_and_run(x, mask):
    class Model(torch.nn.Module):
        def forward(self, x, mask):
            return torch._nested_tensor_from_mask(x, mask)
    traced = torch.jit.trace(Model(), (x, mask))
    ov_model = ov.convert_model(traced, example_input=(x, mask))
    compiled = ov.compile_model(ov_model, "CPU")
    result = compiled({0: x.numpy(), 1: mask.numpy()})
    return result[0]


@pytest.mark.precommit
@pytest.mark.nightly
def test_mixed_padding():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, False],
                         [True, True, False, False]])
    out = convert_and_run(x, mask)
    assert out.shape == (5, 8)
    ref = np.concatenate([x[0, :3].numpy(),
                          x[1, :2].numpy()]).astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=1e-3)


@pytest.mark.precommit
@pytest.mark.nightly
def test_all_valid():
    x = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)
    out = convert_and_run(x, mask)
    assert out.shape == (8, 8)


@pytest.mark.precommit
@pytest.mark.nightly
def test_one_empty_sequence():
    x = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, True, True],
                         [False, False, False, False]])
    out = convert_and_run(x, mask)
    assert out.shape == (4, 8)
    ref = x[0, :4].numpy().astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=1e-3)
