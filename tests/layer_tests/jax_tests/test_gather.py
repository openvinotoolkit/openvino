import pytest
from openvino.tests.layer_tests.jax_tests.utils import run_layer_test

@pytest.mark.parametrize("input_data, indices, axis, expected", [
  
    ([[1, 2, 3], [4, 5, 6]], [0, 2], 1, [[1, 3], [4, 6]]),
  
])
def test_gather(input_data, indices, axis, expected):
    result = run_layer_test("jax.lax.gather", input_data=input_data, indices=indices, axis=axis)
    assert result == expected
