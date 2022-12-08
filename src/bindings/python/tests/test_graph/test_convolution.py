# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.runtime.opset8 as ov


def test_convolution_2d():

    # input_x should have shape N(batch) x C x H x W
    input_x = ov.parameter((1, 1, 9, 9), name="input_x", dtype=np.float32)

    # filter weights should have shape M x C x kH x kW
    input_filter = ov.parameter((1, 1, 3, 3), name="input_filter", dtype=np.float32)

    strides = np.array([1, 1])
    pads_begin = np.array([1, 1])
    pads_end = np.array([1, 1])
    dilations = np.array([1, 1])
    expected_shape = [1, 1, 9, 9]

    # convolution with padding=1 should produce 9 x 9 output:
    node = ov.convolution(input_x, input_filter, strides, pads_begin, pads_end, dilations)
    assert node.get_type_name() == "Convolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape

    # convolution with padding=0 should produce 7 x 7 output:
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])
    expected_shape = [1, 1, 7, 7]

    node = ov.convolution(input_x, input_filter, strides, pads_begin, pads_end, dilations)
    assert node.get_type_name() == "Convolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape

    strides = np.array([2, 2])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])
    expected_shape = [1, 1, 4, 4]

    # convolution with strides=2 should produce 4 x 4 output:
    node = ov.convolution(input_x, input_filter, strides, pads_begin, pads_end, dilations)
    assert node.get_type_name() == "Convolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape

    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([2, 2])
    expected_shape = [1, 1, 5, 5]

    # convolution with dilation=2 should produce 5 x 5 output:
    node = ov.convolution(input_x, input_filter, strides, pads_begin, pads_end, dilations)
    assert node.get_type_name() == "Convolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_convolution_backprop_data():

    output_spatial_shape = [9, 9]
    filter_shape = [1, 1, 3, 3]
    data_shape = [1, 1, 7, 7]
    strides = [1, 1]

    data_node = ov.parameter(shape=data_shape)
    filter_node = ov.parameter(shape=filter_shape)
    output_shape_node = ov.constant(np.array(output_spatial_shape, dtype=np.int64))
    expected_shape = [1, 1, 9, 9]

    deconvolution = ov.convolution_backprop_data(data_node, filter_node, strides, output_shape_node)
    assert deconvolution.get_type_name() == "ConvolutionBackpropData"
    assert deconvolution.get_output_size() == 1
    assert list(deconvolution.get_output_shape(0)) == expected_shape


def test_convolution_v1():
    input_tensor = ov.parameter((1, 1, 16, 16), name="input_tensor", dtype=np.float32)
    filters = ov.parameter((1, 1, 3, 3), name="filters", dtype=np.float32)
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])
    expected_shape = [1, 1, 14, 14]

    node = ov.convolution(input_tensor, filters, strides, pads_begin, pads_end, dilations)
    assert node.get_type_name() == "Convolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
