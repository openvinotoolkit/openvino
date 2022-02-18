# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from tests_compatibility.runtime import get_runtime
from tests_compatibility.test_ngraph.util import run_op_node


def test_lrn():
    input_image_shape = (2, 3, 2, 1)
    input_image = np.arange(int(np.prod(input_image_shape))).reshape(input_image_shape).astype("f")
    axes = np.array([1], dtype=np.int64)
    runtime = get_runtime()
    model = ng.lrn(ng.constant(input_image), ng.constant(axes), alpha=1.0, beta=2.0, bias=1.0, size=3)
    computation = runtime.computation(model)
    result = computation()
    assert np.allclose(
        result,
        np.array(
            [
                [[[0.0], [0.05325444]], [[0.03402646], [0.01869806]], [[0.06805293], [0.03287071]]],
                [[[0.00509002], [0.00356153]], [[0.00174719], [0.0012555]], [[0.00322708], [0.00235574]]],
            ],
            dtype=np.float32,
        ),
    )

    # Test LRN default parameter values
    model = ng.lrn(ng.constant(input_image), ng.constant(axes))
    computation = runtime.computation(model)
    result = computation()
    assert np.allclose(
        result,
        np.array(
            [
                [[[0.0], [0.35355338]], [[0.8944272], [1.0606602]], [[1.7888544], [1.767767]]],
                [[[0.93704253], [0.97827977]], [[1.2493901], [1.2577883]], [[1.5617375], [1.5372968]]],
            ],
            dtype=np.float32,
        ),
    )


def test_lrn_factory():
    alpha = 0.0002
    beta = 0.5
    bias = 2.0
    nsize = 3
    axis = np.array([1], dtype=np.int32)
    x = np.array(
        [
            [
                [
                    [0.31403765, -0.16793324, 1.388258, -0.6902954],
                    [-0.3994045, -0.7833511, -0.30992958, 0.3557573],
                    [-0.4682631, 1.1741459, -2.414789, -0.42783254],
                ],
                [
                    [-0.82199496, -0.03900861, -0.43670088, -0.53810567],
                    [-0.10769883, 0.75242394, -0.2507971, 1.0447186],
                    [-1.4777364, 0.19993274, 0.925649, -2.282516],
                ],
            ]
        ],
        dtype=np.float32,
    )
    excepted = np.array(
        [
            [
                [
                    [0.22205527, -0.11874668, 0.98161197, -0.4881063],
                    [-0.2824208, -0.553902, -0.21915273, 0.2515533],
                    [-0.33109877, 0.8302269, -1.7073234, -0.3024961],
                ],
                [
                    [-0.5812307, -0.02758324, -0.30878326, -0.38049328],
                    [-0.07615435, 0.53203356, -0.17733987, 0.7387126],
                    [-1.0448756, 0.14137045, 0.6544598, -1.6138376],
                ],
            ]
        ],
        dtype=np.float32,
    )
    result = run_op_node([x], ng.lrn, axis, alpha, beta, bias, nsize)

    assert np.allclose(result, excepted)


def test_batch_norm_inference():
    data = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
    gamma = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    beta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    variance = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    epsilon = 9.99e-06
    excepted = np.array([[2.0, 6.0, 12.0], [-2.0, -6.0, -12.0]], dtype=np.float32)

    result = run_op_node([data, gamma, beta, mean, variance], ng.batch_norm_inference, epsilon)

    assert np.allclose(result, excepted)


def test_mvn_no_variance():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                     1, 2, 3, 4, 5, 6, 7, 8, 9,
                     1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32).reshape([1, 3, 3, 3])
    axes = np.array([2, 3], dtype=np.int64)
    epsilon = 1e-9
    normalize_variance = False
    eps_mode = "outside_sqrt"
    excepted = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4,
                         -4, -3, -2, -1, 0, 1, 2, 3, 4,
                         -4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float32).reshape([1, 3, 3, 3])

    result = run_op_node([data], ng.mvn, axes, normalize_variance, epsilon, eps_mode)

    assert np.allclose(result, excepted)


def test_mvn():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                     1, 2, 3, 4, 5, 6, 7, 8, 9,
                     1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32).reshape([1, 3, 3, 3])
    axes = np.array([2, 3], dtype=np.int64)
    epsilon = 1e-9
    normalize_variance = True
    eps_mode = "outside_sqrt"
    excepted = np.array([-1.5491934, -1.161895, -0.7745967,
                         -0.38729835, 0., 0.38729835,
                         0.7745967, 1.161895, 1.5491934,
                         -1.5491934, -1.161895, -0.7745967,
                         -0.38729835, 0., 0.38729835,
                         0.7745967, 1.161895, 1.5491934,
                         -1.5491934, -1.161895, -0.7745967,
                         -0.38729835, 0., 0.38729835,
                         0.7745967, 1.161895, 1.5491934], dtype=np.float32).reshape([1, 3, 3, 3])

    result = run_op_node([data], ng.mvn, axes, normalize_variance, epsilon, eps_mode)

    assert np.allclose(result, excepted)
