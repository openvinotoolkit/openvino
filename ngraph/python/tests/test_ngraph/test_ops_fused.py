# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ngraph as ng
from tests.runtime import get_runtime
from tests import (xfail_issue_36486,
                   xfail_issue_36487,
                   xfail_issue_44976)


def test_elu_operator_with_scalar_and_array():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)
    alpha_value = np.float32(3)

    model = ng.elu(data_value, alpha_value)
    computation = runtime.computation(model)

    result = computation()
    expected = np.array([[-2.9797862, 1.0], [-2.5939941, 3.0]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_elu_operator_with_scalar():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)
    alpha_value = np.float32(3)

    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.elu(parameter_data, alpha_value)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array([[-2.9797862, 1.0], [-2.5939941, 3.0]], dtype=np.float32)
    assert np.allclose(result, expected)


@xfail_issue_44976
def test_fake_quantize():
    runtime = get_runtime()

    data_value = np.arange(24.0, dtype=np.float32).reshape(1, 2, 3, 4)
    input_low_value = np.float32(0)
    input_high_value = np.float32(23)
    output_low_value = np.float32(2)
    output_high_value = np.float32(16)
    levels = np.float32(4)

    data_shape = [1, 2, 3, 4]
    bound_shape = []
    parameter_data = ng.parameter(data_shape, name="data", dtype=np.float32)
    parameter_input_low = ng.parameter(bound_shape, name="input_low", dtype=np.float32)
    parameter_input_high = ng.parameter(bound_shape, name="input_high", dtype=np.float32)
    parameter_output_low = ng.parameter(bound_shape, name="output_low", dtype=np.float32)
    parameter_output_high = ng.parameter(bound_shape, name="output_high", dtype=np.float32)

    model = ng.fake_quantize(
        parameter_data,
        parameter_input_low,
        parameter_input_high,
        parameter_output_low,
        parameter_output_high,
        levels,
    )
    computation = runtime.computation(
        model,
        parameter_data,
        parameter_input_low,
        parameter_input_high,
        parameter_output_low,
        parameter_output_high,
    )

    result = computation(data_value, input_low_value, input_high_value, output_low_value, output_high_value)

    expected = np.array(
        [
            [
                [
                    [
                        [2.0, 2.0, 2.0, 2.0],
                        [6.6666669, 6.6666669, 6.6666669, 6.6666669],
                        [6.6666669, 6.6666669, 6.6666669, 6.6666669],
                    ],
                    [
                        [11.33333301, 11.33333301, 11.33333301, 11.33333301],
                        [11.33333301, 11.33333301, 11.33333301, 11.33333301],
                        [16.0, 16.0, 16.0, 16.0],
                    ],
                ]
            ]
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)


def test_depth_to_space():
    runtime = get_runtime()

    data_value = np.array(
        [
            [
                [[0, 1, 2], [3, 4, 5]],
                [[6, 7, 8], [9, 10, 11]],
                [[12, 13, 14], [15, 16, 17]],
                [[18, 19, 20], [21, 22, 23]],
            ]
        ],
        dtype=np.float32,
    )
    mode = "blocks_first"
    block_size = np.float32(2)

    data_shape = [1, 4, 2, 3]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.depth_to_space(parameter_data, mode, block_size)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array(
        [[[[0, 6, 1, 7, 2, 8], [12, 18, 13, 19, 14, 20], [3, 9, 4, 10, 5, 11], [15, 21, 16, 22, 17, 23]]]],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)


def test_space_to_batch():
    runtime = get_runtime()

    data_value = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.float32)
    data_shape = data_value.shape

    block_shape = np.array([1, 2, 3, 2], dtype=np.int64)
    pads_begin = np.array([0, 0, 1, 0], dtype=np.int64)
    pads_end = np.array([0, 0, 0, 1], dtype=np.int64)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.space_to_batch(parameter_data, block_shape, pads_begin, pads_end)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array(
        [
            [[[0, 0]]],
            [[[0, 0]]],
            [[[0, 2]]],
            [[[1, 0]]],
            [[[3, 5]]],
            [[[4, 0]]],
            [[[0, 0]]],
            [[[0, 0]]],
            [[[6, 8]]],
            [[[7, 0]]],
            [[[9, 11]]],
            [[[10, 0]]],
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)


def test_batch_to_space():
    runtime = get_runtime()

    data = np.array(
        [
            [[[0, 0]]],
            [[[0, 0]]],
            [[[0, 2]]],
            [[[1, 0]]],
            [[[3, 5]]],
            [[[4, 0]]],
            [[[0, 0]]],
            [[[0, 0]]],
            [[[6, 8]]],
            [[[7, 0]]],
            [[[9, 11]]],
            [[[10, 0]]],
        ],
        dtype=np.float32,
    )
    data_shape = data.shape

    block_shape = np.array([1, 2, 3, 2], dtype=np.int64)
    crops_begin = np.array([0, 0, 1, 0], dtype=np.int64)
    crops_end = np.array([0, 0, 0, 1], dtype=np.int64)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.batch_to_space(parameter_data, block_shape, crops_begin, crops_end)
    computation = runtime.computation(model, parameter_data)

    result = computation(data)
    expected = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.float32)

    assert np.allclose(result, expected)


def test_clamp_operator():
    runtime = get_runtime()

    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    min_value = np.float32(3)
    max_value = np.float32(12)

    model = ng.clamp(parameter_data, min_value, max_value)
    computation = runtime.computation(model, parameter_data)

    data_value = np.array([[-5, 9], [45, 3]], dtype=np.float32)

    result = computation(data_value)
    expected = np.clip(data_value, min_value, max_value)
    assert np.allclose(result, expected)


def test_clamp_operator_with_array():
    runtime = get_runtime()

    data_value = np.array([[-5, 9], [45, 3]], dtype=np.float32)
    min_value = np.float32(3)
    max_value = np.float32(12)

    model = ng.clamp(data_value, min_value, max_value)
    computation = runtime.computation(model)

    result = computation()
    expected = np.clip(data_value, min_value, max_value)

    assert np.allclose(result, expected)


def test_squeeze_operator():
    runtime = get_runtime()

    data_shape = [1, 2, 1, 3, 1, 1]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    data_value = np.arange(6.0, dtype=np.float32).reshape([1, 2, 1, 3, 1, 1])
    axes = [2, 4]
    model = ng.squeeze(parameter_data, axes)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.arange(6.0, dtype=np.float32).reshape([1, 2, 3, 1])
    assert np.allclose(result, expected)


def test_squared_difference_operator():
    runtime = get_runtime()

    x1_shape = [1, 2, 3, 4]
    x2_shape = [2, 3, 4]

    parameter_x1 = ng.parameter(x1_shape, name="x1", dtype=np.float32)
    parameter_x2 = ng.parameter(x2_shape, name="x2", dtype=np.float32)

    x1_value = np.arange(24.0, dtype=np.float32).reshape(x1_shape)
    x2_value = np.arange(start=4.0, stop=28.0, step=1.0, dtype=np.float32).reshape(x2_shape)

    model = ng.squared_difference(parameter_x1, parameter_x2)
    computation = runtime.computation(model, parameter_x1, parameter_x2)

    result = computation(x1_value, x2_value)
    expected = np.square(np.subtract(x1_value, x2_value))
    assert np.allclose(result, expected)


def test_shuffle_channels_operator():
    runtime = get_runtime()

    data_shape = [1, 15, 2, 2]
    axis = 1
    groups = 5

    parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)

    data_value = np.arange(60.0, dtype=np.float32).reshape(data_shape)

    model = ng.shuffle_channels(parameter, axis, groups)
    computation = runtime.computation(model, parameter)

    result = computation(data_value)
    expected = np.array(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[12.0, 13.0], [14.0, 15.0]],
                [[24.0, 25.0], [26.0, 27.0]],
                [[36.0, 37.0], [38.0, 39.0]],
                [[48.0, 49.0], [50.0, 51.0]],
                [[4.0, 5.0], [6.0, 7.0]],
                [[16.0, 17.0], [18.0, 19.0]],
                [[28.0, 29.0], [30.0, 31.0]],
                [[40.0, 41.0], [42.0, 43.0]],
                [[52.0, 53.0], [54.0, 55.0]],
                [[8.0, 9.0], [10.0, 11.0]],
                [[20.0, 21.0], [22.0, 23.0]],
                [[32.0, 33.0], [34.0, 35.0]],
                [[44.0, 45.0], [46.0, 47.0]],
                [[56.0, 57.0], [58.0, 59.0]],
            ]
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)


def test_unsqueeze():
    runtime = get_runtime()

    data_shape = [3, 4, 5]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    data_value = np.arange(60.0, dtype=np.float32).reshape(3, 4, 5)
    axes = [0, 4]
    model = ng.unsqueeze(parameter_data, axes)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.arange(60.0, dtype=np.float32).reshape([1, 3, 4, 5, 1])
    assert np.allclose(result, expected)


def test_grn_operator():
    runtime = get_runtime()

    data_value = np.arange(start=1.0, stop=25.0, dtype=np.float32).reshape([1, 2, 3, 4])
    bias = np.float32(1e-6)

    data_shape = [1, 2, 3, 4]

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.grn(parameter_data, bias)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array(
        [
            [
                [
                    [0.0766965, 0.14142136, 0.19611613, 0.24253564],
                    [0.28216633, 0.31622776, 0.34570536, 0.37139067],
                    [0.39391932, 0.41380295, 0.4314555, 0.4472136],
                ],
                [
                    [0.9970545, 0.98994946, 0.9805807, 0.97014254],
                    [0.9593655, 0.9486833, 0.9383431, 0.9284767],
                    [0.91914505, 0.9103665, 0.9021342, 0.8944272],
                ],
            ]
        ],
        dtype=np.float32,
    )

    assert np.allclose(result, expected)


def test_prelu_operator():
    runtime = get_runtime()

    data_shape = [1, 2, 3, 4]
    slope_shape = [2, 3, 1]

    data_value = np.arange(start=1.0, stop=25.0, dtype=np.float32).reshape(data_shape)
    slope_value = np.arange(start=-10.0, stop=-4.0, dtype=np.float32).reshape(slope_shape)
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_slope = ng.parameter(slope_shape, name="Slope", dtype=np.float32)

    model = ng.prelu(parameter_data, parameter_slope)
    computation = runtime.computation(model, parameter_data, parameter_slope)

    result = computation(data_value, slope_value)
    expected = np.clip(data_value, 0, np.inf) + np.clip(data_value, -np.inf, 0) * slope_value
    assert np.allclose(result, expected)


def test_selu_operator():
    runtime = get_runtime()

    data_shape = [4, 2, 3, 1]

    data = np.arange(start=1.0, stop=25.0, dtype=np.float32).reshape(data_shape)
    alpha = np.array(1.6733, dtype=np.float32)
    lambda_value = np.array(1.0507, dtype=np.float32)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    model = ng.selu(parameter_data, alpha, lambda_value)
    computation = runtime.computation(model, parameter_data)

    result = computation(data)
    expected = lambda_value * ((data > 0) * data + (data <= 0) * (alpha * np.exp(data) - alpha))
    assert np.allclose(result, expected)


@xfail_issue_36486
def test_hard_sigmoid_operator():
    runtime = get_runtime()

    data_shape = [3]
    alpha_value = np.float32(0.5)
    beta_value = np.float32(0.6)

    data_value = np.array([-1, 0, 1], dtype=np.float32)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_alpha = ng.parameter([], name="Alpha", dtype=np.float32)
    parameter_beta = ng.parameter([], name="Beta", dtype=np.float32)

    model = ng.hard_sigmoid(parameter_data, parameter_alpha, parameter_beta)
    computation = runtime.computation(model, parameter_data, parameter_alpha, parameter_beta)

    result = computation(data_value, alpha_value, beta_value)
    expected = [0.1, 0.6, 1.0]
    assert np.allclose(result, expected)


@xfail_issue_36487
def test_mvn_operator():
    runtime = get_runtime()

    data_shape = [3, 3, 3, 1]
    across_channels = True
    normalize_variance = True
    eps = np.float32(1e-9)

    data_value = np.array(
        [
            [
                [[0.8439683], [0.5665144], [0.05836735]],
                [[0.02916367], [0.12964272], [0.5060197]],
                [[0.79538304], [0.9411346], [0.9546573]],
            ],
            [
                [[0.17730942], [0.46192095], [0.26480448]],
                [[0.6746842], [0.01665257], [0.62473077]],
                [[0.9240844], [0.9722341], [0.11965699]],
            ],
            [
                [[0.41356155], [0.9129373], [0.59330076]],
                [[0.81929934], [0.7862604], [0.11799799]],
                [[0.69248444], [0.54119414], [0.07513223]],
            ],
        ],
        dtype=np.float32,
    )

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.mvn(parameter_data, across_channels, normalize_variance, eps)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)

    expected = np.array(
        [
            [
                [[0.9951074], [0.14548765], [-1.410561]],
                [[-1.4999886], [-1.1923014], [-0.03975919]],
                [[0.8463296], [1.2926502], [1.3340596]],
            ],
            [
                [[-1.0463363], [-0.1747985], [-0.7784088]],
                [[0.47672555], [-1.5383], [0.32375798]],
                [[1.2404392], [1.3878832], [-1.2228798]],
            ],
            [
                [[-0.3228847], [1.2063044], [0.22751297]],
                [[0.91956615], [0.81839436], [-1.2279599]],
                [[0.5312334], [0.067952], [-1.3592235]],
            ],
        ],
    )

    assert np.allclose(result, expected)


def test_space_to_depth_operator():
    runtime = get_runtime()

    data_shape = [1, 2, 4, 4]
    data_value = np.arange(start=0, stop=32, step=1.0, dtype=np.float32).reshape(data_shape)
    mode = "blocks_first"
    block_size = 2

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.space_to_depth(parameter_data, mode, block_size)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array(
        [
            0,
            2,
            8,
            10,
            16,
            18,
            24,
            26,
            1,
            3,
            9,
            11,
            17,
            19,
            25,
            27,
            4,
            6,
            12,
            14,
            20,
            22,
            28,
            30,
            5,
            7,
            13,
            15,
            21,
            23,
            29,
            31,
        ],
        dtype=np.float32,
    ).reshape(1, 8, 2, 2)
    assert np.allclose(result, expected)

    batch_size = 2
    input_size = 3
    hidden_size = 3

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    W_shape = [hidden_size, input_size]
    R_shape = [hidden_size, hidden_size]
    B_shape = [hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=np.float32)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=np.float32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=np.float32)
    parameter_R = ng.parameter(R_shape, name="R", dtype=np.float32)
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    X_value = np.array(
        [0.3432185, 0.612268, 0.20272376, 0.9513413, 0.30585995, 0.7265472], dtype=np.float32
    ).reshape(X_shape)
    H_t_value = np.array(
        [0.12444675, 0.52055854, 0.46489045, 0.4983964, 0.7730452, 0.28439692], dtype=np.float32
    ).reshape(H_t_shape)
    W_value = np.array(
        [
            0.41930267,
            0.7872176,
            0.89940447,
            0.23659843,
            0.24676207,
            0.17101714,
            0.3147149,
            0.6555601,
            0.4559603,
        ],
        dtype=np.float32,
    ).reshape(W_shape)
    R_value = np.array(
        [
            0.8374871,
            0.86660194,
            0.82114047,
            0.71549815,
            0.18775631,
            0.3182116,
            0.25392973,
            0.38301638,
            0.85531586,
        ],
        dtype=np.float32,
    ).reshape(R_shape)
    B_value = np.array([1.0289404, 1.6362579, 0.4370661], dtype=np.float32).reshape(B_shape)
    activations = ["sigmoid"]
    activation_alpha = []
    activation_beta = []
    clip = 2.88

    model = ng.rnn_cell(
        parameter_X,
        parameter_H_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )
    computation = runtime.computation(
        model, parameter_X, parameter_H_t, parameter_W, parameter_R, parameter_B
    )

    result = computation(X_value, H_t_value, W_value, R_value, B_value)
    expected = np.array(
        [0.94126844, 0.9036043, 0.841243, 0.9468489, 0.934215, 0.873708], dtype=np.float32
    ).reshape(batch_size, hidden_size)

    assert np.allclose(result, expected)


def test_group_convolution_operator():
    runtime = get_runtime()

    data_shape = [1, 4, 2, 2]
    filters_shape = [2, 1, 2, 1, 1]

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_filters = ng.parameter(filters_shape, name="Filters", dtype=np.float32)

    data_value = np.arange(start=1.0, stop=17.0, dtype=np.float32).reshape(data_shape)
    filters_value = np.arange(start=1.0, stop=5.0, dtype=np.float32).reshape(filters_shape)
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ng.group_convolution(parameter_data, parameter_filters, strides, pads_begin, pads_end, dilations)
    computation = runtime.computation(model, parameter_data, parameter_filters)
    result = computation(data_value, filters_value)

    expected = np.array([11, 14, 17, 20, 79, 86, 93, 100], dtype=np.float32).reshape(1, 2, 2, 2)

    assert np.allclose(result, expected)


@pytest.mark.xfail(reason="Computation mismatch")
def test_group_convolution_backprop_data():
    runtime = get_runtime()

    data_shape = [1, 1, 3, 3]
    filters_shape = [1, 1, 1, 3, 3]
    strides = [2, 2]
    output_padding = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]

    data_node = ng.parameter(data_shape, name="Data", dtype=np.float32)
    filters_node = ng.parameter(filters_shape, name="Filters", dtype=np.float32)
    model = ng.group_convolution_backprop_data(
        data_node, filters_node, strides, None, pads_begin, pads_end, output_padding=output_padding
    )

    data_value = np.array(
        [
            0.16857791,
            -0.15161794,
            0.08540368,
            0.1820628,
            -0.21746576,
            0.08245695,
            0.1431433,
            -0.43156421,
            0.30591947,
        ],
        dtype=np.float32,
    ).reshape(data_shape)

    filters_value = np.array(
        [
            -0.06230065,
            0.37932432,
            -0.25388849,
            0.33878803,
            0.43709868,
            -0.22477469,
            0.04118127,
            -0.44696793,
            0.06373066,
        ],
        dtype=np.float32,
    ).reshape(filters_shape)

    computation = runtime.computation(model, data_node, filters_node)
    result = computation(data_value, filters_value)

    expected = np.array(
        [
            0.07368518,
            -0.08925839,
            -0.06627201,
            0.06301362,
            0.03732984,
            -0.01919658,
            -0.00628807,
            -0.02817563,
            -0.01472169,
            0.04392925,
            -0.00689478,
            -0.01549204,
            0.07957941,
            -0.11459791,
            -0.09505399,
            0.07681622,
            0.03604182,
            -0.01853423,
            -0.0270785,
            -0.00680824,
            -0.06650258,
            0.08004665,
            0.07918708,
            0.0724144,
            0.06256775,
            -0.17838378,
            -0.18863615,
            0.20064656,
            0.133717,
            -0.06876295,
            -0.06398046,
            -0.00864975,
            0.19289537,
            -0.01490572,
            -0.13673618,
            0.01949645,
        ],
        dtype=np.float32,
    ).reshape(1, 1, 6, 6)

    assert np.allclose(result, expected)


def test_group_convolution_backprop_data_output_shape():
    runtime = get_runtime()

    data_shape = [1, 1, 1, 10]
    filters_shape = [1, 1, 1, 1, 5]
    strides = [1, 1]

    data_node = ng.parameter(data_shape, name="Data", dtype=np.float32)
    filters_node = ng.parameter(filters_shape, name="Filters", dtype=np.float32)
    output_shape_node = ng.constant(np.array([1, 14], dtype=np.int64))

    model = ng.group_convolution_backprop_data(
        data_node, filters_node, strides, output_shape_node, auto_pad="same_upper"
    )

    data_value = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32).reshape(
        data_shape
    )

    filters_value = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32).reshape(filters_shape)

    computation = runtime.computation(model, data_node, filters_node)
    result = computation(data_value, filters_value)

    expected = np.array(
        [0.0, 1.0, 4.0, 10.0, 18.0, 27.0, 36.0, 45.0, 54.0, 63.0, 62.0, 50.0, 26.0, 9.0], dtype=np.float32,
    ).reshape(1, 1, 1, 14)

    assert np.allclose(result, expected)
