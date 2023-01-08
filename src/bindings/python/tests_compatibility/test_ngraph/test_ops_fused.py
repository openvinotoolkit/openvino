# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from ngraph.impl import Type


def test_elu_operator_with_scalar_and_array():
    data_value = ng.parameter((2, 2), name="data_value", dtype=np.float32)
    alpha_value = np.float32(3)

    model = ng.elu(data_value, alpha_value)
    assert model.get_type_name() == "Elu"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == Type.f32


def test_elu_operator_with_scalar():
    parameter_data = ng.parameter([2, 2], name="Data", dtype=np.float32)
    alpha_value = np.float32(3)

    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.elu(parameter_data, alpha_value)
    assert model.get_type_name() == "Elu"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == Type.f32


def test_fake_quantize():
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
    assert model.get_type_name() == "FakeQuantize"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]
    assert model.get_output_element_type(0) == Type.f32


def test_depth_to_space():
    data_shape = [1, 4, 2, 3]
    mode = "blocks_first"
    block_size = np.float32(2)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.depth_to_space(parameter_data, mode, block_size)
    assert model.get_type_name() == "DepthToSpace"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 1, 4, 6]
    assert model.get_output_element_type(0) == Type.f32


def test_space_to_batch():
    data_value = np.array([[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]], dtype=np.float32)
    data_shape = [1, 2, 2, 3]
    data_shape = data_value.shape

    block_shape = np.array([1, 2, 3, 2], dtype=np.int64)
    pads_begin = np.array([0, 0, 1, 0], dtype=np.int64)
    pads_end = np.array([0, 0, 0, 1], dtype=np.int64)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.space_to_batch(parameter_data, block_shape, pads_begin, pads_end)
    assert model.get_type_name() == "SpaceToBatch"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [12, 1, 1, 2]
    assert model.get_output_element_type(0) == Type.f32


def test_batch_to_space():
    data_shape = [12, 1, 1, 2]

    block_shape = np.array([1, 2, 3, 2], dtype=np.int64)
    crops_begin = np.array([0, 0, 1, 0], dtype=np.int64)
    crops_end = np.array([0, 0, 0, 1], dtype=np.int64)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.batch_to_space(parameter_data, block_shape, crops_begin, crops_end)
    assert model.get_type_name() == "BatchToSpace"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 2, 3]
    assert model.get_output_element_type(0) == Type.f32


def test_clamp_operator():
    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    min_value = np.float32(3)
    max_value = np.float32(12)

    model = ng.clamp(parameter_data, min_value, max_value)
    assert model.get_type_name() == "Clamp"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == Type.f32


def test_clamp_operator_with_array():
    data_value = np.array([[-5, 9], [45, 3]], dtype=np.float32)
    min_value = np.float32(3)
    max_value = np.float32(12)

    model = ng.clamp(data_value, min_value, max_value)
    assert model.get_type_name() == "Clamp"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == Type.f32


def test_squeeze_operator():
    data_shape = [1, 2, 1, 3, 1, 1]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    axes = [2, 4]
    model = ng.squeeze(parameter_data, axes)
    assert model.get_type_name() == "Squeeze"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 3, 1]
    assert model.get_output_element_type(0) == Type.f32


def test_squared_difference_operator():
    x1_shape = [1, 2, 3, 4]
    x2_shape = [2, 3, 4]

    parameter_x1 = ng.parameter(x1_shape, name="x1", dtype=np.float32)
    parameter_x2 = ng.parameter(x2_shape, name="x2", dtype=np.float32)

    model = ng.squared_difference(parameter_x1, parameter_x2)
    assert model.get_type_name() == "SquaredDifference"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]


def test_shuffle_channels_operator():
    data_shape = [1, 15, 2, 2]
    axis = 1
    groups = 5

    parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.shuffle_channels(parameter, axis, groups)
    assert model.get_type_name() == "ShuffleChannels"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 15, 2, 2]


def test_unsqueeze():

    data_shape = [3, 4, 5]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    axes = [0, 4]
    model = ng.unsqueeze(parameter_data, axes)
    assert model.get_type_name() == "Unsqueeze"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 3, 4, 5, 1]


def test_grn_operator():
    bias = np.float32(1e-6)

    data_shape = [1, 2, 3, 4]

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.grn(parameter_data, bias)
    assert model.get_type_name() == "GRN"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == data_shape


def test_prelu_operator():
    data_shape = [1, 2, 3, 4]
    slope_shape = [2, 3, 1]

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_slope = ng.parameter(slope_shape, name="Slope", dtype=np.float32)

    model = ng.prelu(parameter_data, parameter_slope)
    assert model.get_type_name() == "PRelu"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]


def test_selu_operator():
    data_shape = [4, 2, 3, 1]

    alpha = np.array(1.6733, dtype=np.float32)
    lambda_value = np.array(1.0507, dtype=np.float32)

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    model = ng.selu(parameter_data, alpha, lambda_value)
    assert model.get_type_name() == "Selu"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [4, 2, 3, 1]


def test_hard_sigmoid_operator():
    data_shape = [3]

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_alpha = ng.parameter([], name="Alpha", dtype=np.float32)
    parameter_beta = ng.parameter([], name="Beta", dtype=np.float32)

    model = ng.hard_sigmoid(parameter_data, parameter_alpha, parameter_beta)
    assert model.get_type_name() == "HardSigmoid"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [3]


def test_mvn_operator():
    data_shape = [3, 3, 3, 1]
    axes = [0, 2, 3]
    normalize_variance = True
    eps = np.float32(1e-9)
    eps_mode = "outside_sqrt"

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.mvn(parameter_data, axes, normalize_variance, eps, eps_mode)
    assert model.get_type_name() == "MVN"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == data_shape


def test_space_to_depth_operator():
    data_shape = [1, 2, 4, 4]
    mode = "blocks_first"
    block_size = 2

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.space_to_depth(parameter_data, mode, block_size)
    assert model.get_type_name() == "SpaceToDepth"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 8, 2, 2]

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
    assert model.get_type_name() == "RNNCell"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [batch_size, hidden_size]


def test_group_convolution_operator():
    data_shape = [1, 4, 2, 2]
    filters_shape = [2, 1, 2, 1, 1]

    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_filters = ng.parameter(filters_shape, name="Filters", dtype=np.float32)

    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ng.group_convolution(parameter_data, parameter_filters, strides, pads_begin, pads_end, dilations)
    assert model.get_type_name() == "GroupConvolution"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 2, 2, 2]


def test_group_convolution_backprop_data():
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

    assert model.get_type_name() == "GroupConvolutionBackpropData"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 1, 6, 6]


def test_group_convolution_backprop_data_output_shape():
    data_shape = [1, 1, 1, 10]
    filters_shape = [1, 1, 1, 1, 5]
    strides = [1, 1]

    data_node = ng.parameter(data_shape, name="Data", dtype=np.float32)
    filters_node = ng.parameter(filters_shape, name="Filters", dtype=np.float32)
    output_shape_node = ng.constant(np.array([1, 14], dtype=np.int64))

    model = ng.group_convolution_backprop_data(
        data_node, filters_node, strides, output_shape_node, auto_pad="same_upper"
    )
    assert model.get_type_name() == "GroupConvolutionBackpropData"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 1, 1, 14]
