# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset8 as ov
from openvino import Type


@pytest.fixture()
def ndarray_1x1x4x4():
    return np.arange(11, 27, dtype=np.float32).reshape(1, 1, 4, 4)


def test_avg_pool_2d(ndarray_1x1x4x4):
    input_data = ndarray_1x1x4x4
    param = ov.parameter(input_data.shape, name="A", dtype=np.float32)

    kernel_shape = [2, 2]
    spatial_dim_count = len(kernel_shape)
    pads_begin = [0] * spatial_dim_count
    pads_end = [0] * spatial_dim_count
    strides = [2, 2]
    exclude_pad = True

    node = ov.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    assert node.get_type_name() == "AvgPool"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 1, 2, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_avg_pooling_3d(ndarray_1x1x4x4):
    data = ndarray_1x1x4x4
    data = np.broadcast_to(data, (1, 1, 4, 4, 4))
    param = ov.parameter(list(data.shape))
    kernel_shape = [2, 2, 2]
    strides = [2, 2, 2]
    spatial_dim_count = len(kernel_shape)
    pads_begin = [0] * spatial_dim_count
    pads_end = [0] * spatial_dim_count
    exclude_pad = True

    node = ov.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    assert node.get_type_name() == "AvgPool"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 1, 2, 2, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_max_pool_basic():
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [2, 2]
    rounding_type = "floor"
    auto_pad = None
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    node = ov.max_pool(
        data_node,
        strides,
        dilations,
        pads_begin,
        pads_end,
        kernel_shape,
        rounding_type,
        auto_pad,
        index_et,
    )
    assert node.get_type_name() == "MaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [1, 1, 3, 3]
    assert list(node.get_output_shape(1)) == [1, 1, 3, 3]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32


def test_max_pool_strides():
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [2, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [2, 2]
    rounding_type = "floor"
    auto_pad = None
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    node = ov.max_pool(
        data_node,
        strides,
        dilations,
        pads_begin,
        pads_end,
        kernel_shape,
        rounding_type,
        auto_pad,
        index_et,
    )
    assert node.get_type_name() == "MaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [1, 1, 2, 3]
    assert list(node.get_output_shape(1)) == [1, 1, 2, 3]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32


def test_max_pool_kernel_shape1x1():
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [1, 1]
    rounding_type = "floor"
    auto_pad = None
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    node = ov.max_pool(
        data_node,
        strides,
        dilations,
        pads_begin,
        pads_end,
        kernel_shape,
        rounding_type,
        auto_pad,
        index_et,
    )
    assert node.get_type_name() == "MaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [1, 1, 4, 4]
    assert list(node.get_output_shape(1)) == [1, 1, 4, 4]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32


def test_max_pool_kernel_shape3x3():
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [3, 3]
    rounding_type = "floor"
    auto_pad = None
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    node = ov.max_pool(
        data_node,
        strides,
        dilations,
        pads_begin,
        pads_end,
        kernel_shape,
        rounding_type,
        auto_pad,
        index_et,
    )
    assert node.get_type_name() == "MaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [1, 1, 2, 2]
    assert list(node.get_output_shape(1)) == [1, 1, 2, 2]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32


def test_max_pool_non_zero_pads():
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    kernel_shape = [2, 2]
    rounding_type = "floor"
    auto_pad = None
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    node = ov.max_pool(
        data_node,
        strides,
        dilations,
        pads_begin,
        pads_end,
        kernel_shape,
        rounding_type,
        auto_pad,
        index_et,
    )
    assert node.get_type_name() == "MaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [1, 1, 5, 5]
    assert list(node.get_output_shape(1)) == [1, 1, 5, 5]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32


def test_max_pool_same_upper_auto_pads():
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [2, 2]
    auto_pad = "same_upper"
    rounding_type = "floor"
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    node = ov.max_pool(
        data_node,
        strides,
        dilations,
        pads_begin,
        pads_end,
        kernel_shape,
        rounding_type,
        auto_pad,
        index_et,
    )
    assert node.get_type_name() == "MaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [1, 1, 4, 4]
    assert list(node.get_output_shape(1)) == [1, 1, 4, 4]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32


def test_max_pool_same_lower_auto_pads():
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [2, 2]
    auto_pad = "same_lower"
    rounding_type = "floor"
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    node = ov.max_pool(
        data_node,
        strides,
        dilations,
        pads_begin,
        pads_end,
        kernel_shape,
        rounding_type,
        auto_pad,
        index_et,
    )
    assert node.get_type_name() == "MaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [1, 1, 4, 4]
    assert list(node.get_output_shape(1)) == [1, 1, 4, 4]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32
