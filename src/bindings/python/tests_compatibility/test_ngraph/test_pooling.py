# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Type


@pytest.fixture
def _ndarray_1x1x4x4():
    return np.arange(11, 27, dtype=np.float32).reshape(1, 1, 4, 4)


def test_avg_pool_2d(_ndarray_1x1x4x4):
    input_data = _ndarray_1x1x4x4
    param = ng.parameter(input_data.shape, name="A", dtype=np.float32)

    kernel_shape = [2, 2]
    spatial_dim_count = len(kernel_shape)
    pads_begin = [0] * spatial_dim_count
    pads_end = [0] * spatial_dim_count
    strides = [2, 2]
    exclude_pad = True

    avg_pool_node = ng.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    assert avg_pool_node.get_type_name() == "AvgPool"
    assert avg_pool_node.get_output_size() == 1
    assert list(avg_pool_node.get_output_shape(0)) == [1, 1, 2, 2]
    assert avg_pool_node.get_output_element_type(0) == Type.f32


def test_avg_pooling_3d(_ndarray_1x1x4x4):
    data = _ndarray_1x1x4x4
    data = np.broadcast_to(data, (1, 1, 4, 4, 4))
    param = ng.parameter(list(data.shape))
    kernel_shape = [2, 2, 2]
    strides = [2, 2, 2]
    spatial_dim_count = len(kernel_shape)
    pads_begin = [0] * spatial_dim_count
    pads_end = [0] * spatial_dim_count
    exclude_pad = True

    avgpool = ng.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    assert avgpool.get_type_name() == "AvgPool"
    assert avgpool.get_output_size() == 1
    assert list(avgpool.get_output_shape(0)) == [1, 1, 2, 2, 2]
    assert avgpool.get_output_element_type(0) == Type.f32


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

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ng.max_pool(
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
    assert maxpool_node.get_type_name() == "MaxPool"
    assert maxpool_node.get_output_size() == 2
    assert list(maxpool_node.get_output_shape(0)) == [1, 1, 3, 3]
    assert list(maxpool_node.get_output_shape(1)) == [1, 1, 3, 3]
    assert maxpool_node.get_output_element_type(0) == Type.f32
    assert maxpool_node.get_output_element_type(1) == Type.i32


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

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ng.max_pool(
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
    assert maxpool_node.get_type_name() == "MaxPool"
    assert maxpool_node.get_output_size() == 2
    assert list(maxpool_node.get_output_shape(0)) == [1, 1, 2, 3]
    assert list(maxpool_node.get_output_shape(1)) == [1, 1, 2, 3]
    assert maxpool_node.get_output_element_type(0) == Type.f32
    assert maxpool_node.get_output_element_type(1) == Type.i32


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

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ng.max_pool(
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
    assert maxpool_node.get_type_name() == "MaxPool"
    assert maxpool_node.get_output_size() == 2
    assert list(maxpool_node.get_output_shape(0)) == [1, 1, 4, 4]
    assert list(maxpool_node.get_output_shape(1)) == [1, 1, 4, 4]
    assert maxpool_node.get_output_element_type(0) == Type.f32
    assert maxpool_node.get_output_element_type(1) == Type.i32


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

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ng.max_pool(
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
    assert maxpool_node.get_type_name() == "MaxPool"
    assert maxpool_node.get_output_size() == 2
    assert list(maxpool_node.get_output_shape(0)) == [1, 1, 2, 2]
    assert list(maxpool_node.get_output_shape(1)) == [1, 1, 2, 2]
    assert maxpool_node.get_output_element_type(0) == Type.f32
    assert maxpool_node.get_output_element_type(1) == Type.i32


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

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ng.max_pool(
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
    assert maxpool_node.get_type_name() == "MaxPool"
    assert maxpool_node.get_output_size() == 2
    assert list(maxpool_node.get_output_shape(0)) == [1, 1, 5, 5]
    assert list(maxpool_node.get_output_shape(1)) == [1, 1, 5, 5]
    assert maxpool_node.get_output_element_type(0) == Type.f32
    assert maxpool_node.get_output_element_type(1) == Type.i32


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

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ng.max_pool(
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
    assert maxpool_node.get_type_name() == "MaxPool"
    assert maxpool_node.get_output_size() == 2
    assert list(maxpool_node.get_output_shape(0)) == [1, 1, 4, 4]
    assert list(maxpool_node.get_output_shape(1)) == [1, 1, 4, 4]
    assert maxpool_node.get_output_element_type(0) == Type.f32
    assert maxpool_node.get_output_element_type(1) == Type.i32


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

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ng.max_pool(
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
    assert maxpool_node.get_type_name() == "MaxPool"
    assert maxpool_node.get_output_size() == 2
    assert list(maxpool_node.get_output_shape(0)) == [1, 1, 4, 4]
    assert list(maxpool_node.get_output_shape(1)) == [1, 1, 4, 4]
    assert maxpool_node.get_output_element_type(0) == Type.f32
    assert maxpool_node.get_output_element_type(1) == Type.i32
