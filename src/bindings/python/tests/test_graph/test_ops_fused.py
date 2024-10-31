# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset8 as ov
from openvino import Type


def test_elu_operator_with_scalar_and_array():
    data_value = ov.parameter((2, 2), name="data_value", dtype=np.float32)
    alpha_value = np.float32(3)

    model = ov.elu(data_value, alpha_value)

    assert model.get_type_name() == "Elu"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]


def test_elu_operator_with_scalar():
    alpha_value = np.float32(3)
    parameter_data = ov.parameter([2, 2], name="Data", dtype=np.float32)

    model = ov.elu(parameter_data, alpha_value)

    assert model.get_type_name() == "Elu"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]


def test_fake_quantize():
    levels = np.int32(4)

    data_shape = [1, 2, 3, 4]
    bound_shape = []
    parameter_data = ov.parameter(data_shape, name="data", dtype=np.float32)
    parameter_input_low = ov.parameter(bound_shape, name="input_low", dtype=np.float32)
    parameter_input_high = ov.parameter(bound_shape, name="input_high", dtype=np.float32)
    parameter_output_low = ov.parameter(bound_shape, name="output_low", dtype=np.float32)
    parameter_output_high = ov.parameter(bound_shape, name="output_high", dtype=np.float32)

    model = ov.fake_quantize(
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


def test_depth_to_space():
    mode = "blocks_first"
    block_size = np.int32(2)
    data_shape = [1, 4, 2, 3]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.depth_to_space(parameter_data, mode, block_size)

    assert model.get_type_name() == "DepthToSpace"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 1, 4, 6]


def test_space_to_batch():
    data_shape = [1, 2, 2, 3]
    block_shape = np.array([1, 2, 3, 2], dtype=np.int64)
    pads_begin = np.array([0, 0, 1, 0], dtype=np.int64)
    pads_end = np.array([0, 0, 0, 1], dtype=np.int64)
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.space_to_batch(parameter_data, block_shape, pads_begin, pads_end)

    assert model.get_type_name() == "SpaceToBatch"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [12, 1, 1, 2]


def test_batch_to_space():
    data_shape = [12, 1, 1, 2]
    block_shape = np.array([1, 2, 3, 2], dtype=np.int64)
    crops_begin = np.array([0, 0, 1, 0], dtype=np.int64)
    crops_end = np.array([0, 0, 0, 1], dtype=np.int64)
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.batch_to_space(parameter_data, block_shape, crops_begin, crops_end)

    assert model.get_type_name() == "BatchToSpace"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 2, 2, 3]


def test_clamp_operator():
    data_shape = [2, 2]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)
    min_value = np.float32(3)
    max_value = np.float32(12)

    model = ov.clamp(parameter_data, min_value, max_value)

    assert model.get_type_name() == "Clamp"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]


def test_squared_difference_operator():
    x1_shape = [1, 2, 3, 4]
    x2_shape = [2, 3, 4]

    parameter_x1 = ov.parameter(x1_shape, name="x1", dtype=np.float32)
    parameter_x2 = ov.parameter(x2_shape, name="x2", dtype=np.float32)

    model = ov.squared_difference(parameter_x1, parameter_x2)
    assert model.get_type_name() == "SquaredDifference"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]


def test_shuffle_channels_operator():
    data_shape = [1, 15, 2, 2]
    axis = 1
    groups = 5
    parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.shuffle_channels(parameter, axis, groups)
    assert model.get_type_name() == "ShuffleChannels"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 15, 2, 2]


def test_unsqueeze():
    data_shape = [3, 4, 5]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)
    axes = [0, 4]

    model = ov.unsqueeze(parameter_data, axes)
    assert model.get_type_name() == "Unsqueeze"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 3, 4, 5, 1]


def test_grn_operator():
    bias = np.float32(1e-6)
    data_shape = [1, 2, 3, 4]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.grn(parameter_data, bias)
    assert model.get_type_name() == "GRN"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 2, 3, 4]


def test_prelu_operator():
    data_shape = [1, 2, 3, 4]
    slope_shape = [2, 3, 1]
    data_value = np.arange(start=1.0, stop=25.0, dtype=np.float32).reshape(data_shape)
    slope_value = np.arange(start=-10.0, stop=-4.0, dtype=np.float32).reshape(slope_shape)
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_slope = ov.parameter(slope_shape, name="Slope", dtype=np.float32)

    model = ov.prelu(parameter_data, parameter_slope)
    expected = np.clip(data_value, 0, np.inf) + np.clip(data_value, -np.inf, 0) * slope_value
    assert model.get_type_name() == "PRelu"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == list(expected.shape)


def test_selu_operator():
    data_shape = [4, 2, 3, 1]
    alpha = np.array(1.6733, dtype=np.float32)
    lambda_value = np.array(1.0507, dtype=np.float32)

    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)
    model = ov.selu(parameter_data, alpha, lambda_value)
    assert model.get_type_name() == "Selu"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [4, 2, 3, 1]


def test_hard_sigmoid_operator():
    data_shape = [3]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_alpha = ov.parameter([], name="Alpha", dtype=np.float32)
    parameter_beta = ov.parameter([], name="Beta", dtype=np.float32)

    model = ov.hard_sigmoid(parameter_data, parameter_alpha, parameter_beta)
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
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.mvn(parameter_data, axes, normalize_variance, eps, eps_mode)
    assert model.get_type_name() == "MVN"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [3, 3, 3, 1]


@pytest.mark.skip(reason="Sporadically failed. Need further investigation. Ticket - 95970")
def test_space_to_depth_operator():
    data_shape = [1, 2, 4, 4]
    mode = "blocks_first"
    block_size = 2

    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.space_to_depth(parameter_data, mode, block_size)
    assert model.get_type_name() == "SpaceToDepth"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 8, 2, 2]

    batch_size = 2
    input_size = 3
    hidden_size = 3

    x_shape = [batch_size, input_size]
    h_t_shape = [batch_size, hidden_size]
    w_shape = [hidden_size, input_size]
    r_shape = [hidden_size, hidden_size]
    b_shape = [hidden_size]

    parameter_x = ov.parameter(x_shape, name="X", dtype=np.float32)
    parameter_h_t = ov.parameter(h_t_shape, name="H_t", dtype=np.float32)
    parameter_w = ov.parameter(w_shape, name="W", dtype=np.float32)
    parameter_r = ov.parameter(r_shape, name="R", dtype=np.float32)
    parameter_b = ov.parameter(b_shape, name="B", dtype=np.float32)

    activations = ["sigmoid"]
    activation_alpha = []
    activation_beta = []
    clip = 2.88

    model = ov.rnn_cell(
        parameter_x,
        parameter_h_t,
        parameter_w,
        parameter_r,
        parameter_b,
        hidden_size,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )
    assert model.get_type_name() == "SpaceToDepth"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [batch_size, hidden_size]


def test_group_convolution_operator():
    data_shape = [1, 4, 2, 2]
    filters_shape = [2, 1, 2, 1, 1]

    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)
    parameter_filters = ov.parameter(filters_shape, name="Filters", dtype=np.float32)

    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ov.group_convolution(parameter_data, parameter_filters, strides, pads_begin, pads_end, dilations)
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

    data_node = ov.parameter(data_shape, name="Data", dtype=np.float32)
    filters_node = ov.parameter(filters_shape, name="Filters", dtype=np.float32)
    model = ov.group_convolution_backprop_data(
        data_node, filters_node, strides, None, pads_begin, pads_end, output_padding=output_padding,
    )

    assert model.get_type_name() == "GroupConvolutionBackpropData"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 1, 6, 6]


def test_group_convolution_backprop_data_output_shape():
    data_shape = [1, 1, 1, 10]
    filters_shape = [1, 1, 1, 1, 5]
    strides = [1, 1]

    data_node = ov.parameter(data_shape, name="Data", dtype=np.float32)
    filters_node = ov.parameter(filters_shape, name="Filters", dtype=np.float32)
    output_shape_node = ov.constant(np.array([1, 14], dtype=np.int64))

    model = ov.group_convolution_backprop_data(
        data_node, filters_node, strides, output_shape_node, auto_pad="same_upper",
    )

    assert model.get_type_name() == "GroupConvolutionBackpropData"
    assert model.get_output_size() == 1
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [1, 1, 1, 14]
