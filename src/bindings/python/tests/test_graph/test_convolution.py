# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino import Type
import openvino.opset8 as ov


@pytest.mark.parametrize(("strides", "pads_begin", "pads_end", "dilations", "expected_shape"), [
    (np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), [1, 1, 9, 9]),
    (np.array([1, 1]), np.array([0, 0]), np.array([0, 0]), np.array([1, 1]), [1, 1, 7, 7]),
    (np.array([2, 2]), np.array([0, 0]), np.array([0, 0]), np.array([1, 1]), [1, 1, 4, 4]),
    (np.array([1, 1]), np.array([0, 0]), np.array([0, 0]), np.array([2, 2]), [1, 1, 5, 5]),
])
@pytest.mark.parametrize("op_name", ["Convolution", "convolution", "ConvolutionOpset8"])
def test_convolution_2d(strides, pads_begin, pads_end, dilations, expected_shape, op_name):

    # input_x should have shape N(batch) x C x H x W
    input_x = ov.parameter((1, 1, 9, 9), name="input_x", dtype=np.float32)

    # filter weights should have shape M x C x kH x kW
    input_filter = ov.parameter((1, 1, 3, 3), name="input_filter", dtype=np.float32)

    node = ov.convolution(input_x, input_filter, strides, pads_begin, pads_end, dilations, name=op_name)
    assert node.get_type_name() == "Convolution"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize("op_name", ["Convolution", "convolution", "ConvolutionOpset8"])
def test_convolution_backprop_data(op_name):

    output_spatial_shape = [9, 9]
    filter_shape = [1, 1, 3, 3]
    data_shape = [1, 1, 7, 7]
    strides = [1, 1]

    data_node = ov.parameter(shape=data_shape)
    filter_node = ov.parameter(shape=filter_shape)
    output_shape_node = ov.constant(np.array(output_spatial_shape, dtype=np.int64))

    deconvolution = ov.convolution_backprop_data(data_node, filter_node, strides, output_shape_node, name=op_name)
    assert deconvolution.get_type_name() == "ConvolutionBackpropData"
    assert deconvolution.get_friendly_name() == op_name
    assert deconvolution.get_output_size() == 1
    assert list(deconvolution.get_output_shape(0)) == [1, 1, 9, 9]
    assert deconvolution.get_output_element_type(0) == Type.f32
