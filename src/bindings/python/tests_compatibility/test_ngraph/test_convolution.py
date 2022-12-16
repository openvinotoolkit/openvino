# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Type


@pytest.mark.parametrize(("strides", "pads_begin", "pads_end", "dilations", "expected_shape"), [
    (np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1]), [1, 1, 9, 9]),
    (np.array([1, 1]), np.array([0, 0]), np.array([0, 0]), np.array([1, 1]), [1, 1, 7, 7]),
    (np.array([2, 2]), np.array([0, 0]), np.array([0, 0]), np.array([1, 1]), [1, 1, 4, 4]),
    (np.array([1, 1]), np.array([0, 0]), np.array([0, 0]), np.array([2, 2]), [1, 1, 5, 5]),
])
def test_convolution_2d(strides, pads_begin, pads_end, dilations, expected_shape):

    # input_x should have shape N(batch) x C x H x W
    input_x = ng.parameter((1, 1, 9, 9), name="input_data", dtype=np.float32)

    # filter weights should have shape M x C x kH x kW
    input_filter = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32).reshape(
        1, 1, 3, 3
    )

    node = ng.convolution(input_x, input_filter, strides, pads_begin, pads_end, dilations)

    assert node.get_type_name() == "Convolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


def test_convolution_backprop_data():
    output_spatial_shape = [9, 9]
    filter_shape = [1, 1, 3, 3]
    data_shape = [1, 1, 7, 7]
    strides = [1, 1]

    data_node = ng.parameter(shape=data_shape)
    filter_node = ng.parameter(shape=filter_shape)
    output_shape_node = ng.constant(np.array(output_spatial_shape, dtype=np.int64))

    deconvolution = ng.convolution_backprop_data(data_node, filter_node, strides, output_shape_node)
    assert deconvolution.get_type_name() == "ConvolutionBackpropData"
    assert deconvolution.get_output_size() == 1
    assert list(deconvolution.get_output_shape(0)) == [1, 1, 9, 9]
    assert deconvolution.get_output_element_type(0) == Type.f32


def test_convolution_v1():
    input_tensor = np.arange(-128, 128, 1, dtype=np.float32).reshape(1, 1, 16, 16)
    filters = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filters[0, 0, 0, 0] = -1
    filters[0, 0, 1, 1] = -1
    filters[0, 0, 2, 2] = -1
    filters[0, 0, 0, 2] = -1
    filters[0, 0, 2, 0] = -1
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])

    node = ng.convolution(input_tensor, filters, strides, pads_begin, pads_end, dilations)

    assert node.get_type_name() == "Convolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 1, 14, 14]
    assert node.get_output_element_type(0) == Type.f32
