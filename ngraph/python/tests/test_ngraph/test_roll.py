import ngraph as ng
import numpy as np
from tests.runtime import get_runtime


def test_roll():
    runtime = get_runtime()
    input = np.reshape(np.arange(10), (2, 5))
    input_tensor = ng.constant(input)
    input_shift = ng.constant(np.array([-10, 7], dtype=np.int32))
    input_axes = ng.constant(np.array([-1, 0], dtype=np.int32))

    roll_node = ng.roll(input_tensor, input_shift, input_axes)
    computation = runtime.computation(roll_node)
    roll_results = computation()
    expected_results = np.roll(input, shift=(-10, 7), axis=(-1, 0))

    assert np.allclose(roll_results, expected_results)
