import ngraph as ng
import numpy as np
from tests import xfail_issue_59935
from tests.runtime import get_runtime


@xfail_issue_59935
def test_adaptive_avg_pool():
    runtime = get_runtime()
    input = np.reshape([0, 4, 1, 3, -2, -5, -2,
                        -2, 1, -3, 1, -3, -4, 0,
                        -2, 1, -1, -2, 3, -1, -3,

                        -1, -2, 3, 4, -3, -4, 1,
                        2, 0, -4, -5, -2, -2, -3,
                        2, 3, 1, -5, 2, -4, -2], (2, 3, 7))
    input_tensor = ng.constant(input)
    output_shape = ng.constant(np.array([3], dtype=np.int32))

    adaptive_pool_node = ng.adaptive_avg_pool(input_tensor, output_shape)
    computation = runtime.computation(adaptive_pool_node)
    adaptive_pool_results = computation()
    expected_results = np.reshape([1.66666663, 0.66666669, -3.,
                                   -1.33333337, -1.66666663, -2.33333325,
                                   -0.66666669, 0., -0.33333334,

                                   0., 1.33333337, -2.,
                                   -0.66666669, -3.66666675, -2.33333325,
                                   2., -0.66666669, -1.33333337], (2, 3, 3))

    assert np.allclose(adaptive_pool_results, expected_results)


@xfail_issue_59935
def test_adaptive_max_pool():
    runtime = get_runtime()
    input = np.reshape([0, 4, 1, 3, -2, -5, -2,
                        -2, 1, -3, 1, -3, -4, 0,
                        -2, 1, -1, -2, 3, -1, -3,

                        -1, -2, 3, 4, -3, -4, 1,
                        2, 0, -4, -5, -2, -2, -3,
                        2, 3, 1, -5, 2, -4, -2], (2, 3, 7))
    input_tensor = ng.constant(input)
    output_shape = ng.constant(np.array([3], dtype=np.int32))

    adaptive_pool_node = ng.adaptive_max_pool(input_tensor, output_shape)
    computation = runtime.computation(adaptive_pool_node)
    adaptive_pool_results = computation()
    expected_results = np.reshape([4, 3, -2,
                                   1, 1, 0,
                                   1, 3, 3,

                                   3, 4, 1,
                                   2, -2, -2,
                                   3, 2, 2], (2, 3, 3))

    expected_indices = np.reshape([1, 3, 4,
                                   1, 3, 6,
                                   1, 4, 4,

                                   2, 3, 6,
                                   0, 4, 4,
                                   1, 4, 4], (2, 3, 3))

    assert np.allclose(adaptive_pool_results, [expected_results, expected_indices])
