# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset8 as ov
from tests.runtime import get_runtime


@pytest.fixture
def _ndarray_1x1x4x4():
    return np.arange(11, 27, dtype=np.float32).reshape(1, 1, 4, 4)


def test_avg_pool_2d(_ndarray_1x1x4x4):
    runtime = get_runtime()
    input_data = _ndarray_1x1x4x4
    param = ov.parameter(input_data.shape, name="A", dtype=np.float32)

    kernel_shape = [2, 2]
    spatial_dim_count = len(kernel_shape)
    pads_begin = [0] * spatial_dim_count
    pads_end = [0] * spatial_dim_count
    strides = [2, 2]
    exclude_pad = True
    expected = [[[[13.5, 15.5], [21.5, 23.5]]]]

    avg_pool_node = ov.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    expected = [[[[13.5, 14.5, 15.5], [17.5, 18.5, 19.5], [21.5, 22.5, 23.5]]]]
    strides = [1, 1]
    avg_pool_node = ov.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    pads_begin = [1, 1]
    pads_end = [1, 1]
    strides = [2, 2]
    exclude_pad = True

    expected = [[[[11.0, 12.5, 14.0], [17.0, 18.5, 20.0], [23.0, 24.5, 26.0]]]]
    avg_pool_node = ov.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    exclude_pad = False
    expected = [[[[2.75, 6.25, 3.5], [8.5, 18.5, 10.0], [5.75, 12.25, 6.5]]]]
    avg_pool_node = ov.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)


def test_avg_pooling_3d(_ndarray_1x1x4x4):
    rt = get_runtime()
    data = _ndarray_1x1x4x4
    data = np.broadcast_to(data, (1, 1, 4, 4, 4))
    param = ov.parameter(list(data.shape))
    kernel_shape = [2, 2, 2]
    strides = [2, 2, 2]
    spatial_dim_count = len(kernel_shape)
    pads_begin = [0] * spatial_dim_count
    pads_end = [0] * spatial_dim_count
    exclude_pad = True

    avgpool = ov.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    comp = rt.computation(avgpool, param)
    result = comp(data)
    result_ref = [[[[[13.5, 15.5], [21.5, 23.5]], [[13.5, 15.5], [21.5, 23.5]]]]]
    assert np.allclose(result, result_ref)


def test_max_pool_basic():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
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
    maxpool_node = ov.max_pool(
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
    comp = rt.computation(maxpool_node, data_node)
    result = comp(data)

    expected = np.array(
        [[[[5.5, 6.5, 7.5], [9.5, 10.5, 11.5], [13.5, 14.5, 15.5]]]], dtype=np.float32
    )
    expected_idx = np.array([[[[5, 6, 7], [9, 10, 11], [13, 14, 15]]]], dtype=np.int32)
    assert np.allclose(result[0], expected)
    assert np.allclose(result[1], expected_idx)


def test_max_pool_strides():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
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
    maxpool_node = ov.max_pool(
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
    comp = rt.computation(maxpool_node, data_node)
    result = comp(data)

    expected = np.array([[[[5.5, 6.5, 7.5], [13.5, 14.5, 15.5]]]], dtype=np.float32)
    expected_idx = np.array([[[[5, 6, 7], [13, 14, 15]]]], dtype=np.int32)
    assert np.allclose(result[0], expected)
    assert np.allclose(result[1], expected_idx)


def test_max_pool_kernel_shape1x1():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
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
    maxpool_node = ov.max_pool(
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
    comp = rt.computation(maxpool_node, data_node)
    result = comp(data)

    assert np.allclose(result[0], data)
    assert np.allclose(result[1], np.arange(0, 16, dtype=np.int32).reshape((1, 1, 4, 4)))


def test_max_pool_kernel_shape3x3():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
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
    maxpool_node = ov.max_pool(
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
    comp = rt.computation(maxpool_node, data_node)
    result = comp(data)

    expected = np.array([[[[10.5, 11.5], [14.5, 15.5]]]], dtype=np.float32)
    assert np.allclose(result[0], expected)


def test_max_pool_non_zero_pads():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    #  0   0  ,  0  ,  0  ,  0,    0
    #  0 [ 0.5,  1.5,  2.5,  3.5], 0,
    #  0 [ 4.5,  5.5,  6.5,  7.5], 0,
    #  0 [ 8.5,  9.5, 10.5, 11.5], 0,
    #  0 [12.5, 13.5, 14.5, 15.5], 0
    #  0   0  ,  0  ,  0  ,  0,    0
    kernel_shape = [2, 2]
    rounding_type = "floor"
    auto_pad = None
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ov.max_pool(
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
    comp = rt.computation(maxpool_node, data_node)
    result = comp(data)

    expected = np.array(
        [
            [
                [
                    [0.5, 1.5, 2.5, 3.5, 3.5],
                    [4.5, 5.5, 6.5, 7.5, 7.5],
                    [8.5, 9.5, 10.5, 11.5, 11.5],
                    [12.5, 13.5, 14.5, 15.5, 15.5],
                    [12.5, 13.5, 14.5, 15.5, 15.5],
                ]
            ]
        ],
        dtype=np.float32,
    )
    expected_idx = np.array(
        [
            [
                [
                    [0, 1, 2, 3, 3],
                    [4, 5, 6, 7, 7],
                    [8, 9, 10, 11, 11],
                    [12, 13, 14, 15, 15],
                    [12, 13, 14, 15, 15],
                ]
            ]
        ],
        dtype=np.int32,
    )
    assert np.allclose(result[0], expected)
    assert np.allclose(result[1], expected_idx)


def test_max_pool_same_upper_auto_pads():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    # [ 0.5,  1.5,  2.5,  3.5], 0,
    # [ 4.5,  5.5,  6.5,  7.5], 0,
    # [ 8.5,  9.5, 10.5, 11.5], 0,
    # [12.5, 13.5, 14.5, 15.5], 0
    #   0  ,  0  ,  0  ,  0,    0
    kernel_shape = [2, 2]
    auto_pad = "same_upper"
    rounding_type = "floor"
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ov.max_pool(
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
    comp = rt.computation(maxpool_node, data_node)
    result = comp(data)

    expected = np.array(
        [
            [
                [
                    [5.5, 6.5, 7.5, 7.5],
                    [9.5, 10.5, 11.5, 11.5],
                    [13.5, 14.5, 15.5, 15.5],
                    [13.5, 14.5, 15.5, 15.5],
                ]
            ]
        ],
        dtype=np.float32,
    )
    expected_idx = np.array(
        [
            [
                [
                    [5, 6, 7, 7],
                    [9, 10, 11, 11],
                    [13, 14, 15, 15],
                    [13, 14, 15, 15],
                ]
            ]
        ],
        dtype=np.int32,
    )
    assert np.allclose(result[0], expected)
    assert np.allclose(result[1], expected_idx)


def test_max_pool_same_lower_auto_pads():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    #  0   0  ,  0  ,  0  ,  0,
    #  0 [ 0.5,  1.5,  2.5,  3.5],
    #  0 [ 4.5,  5.5,  6.5,  7.5],
    #  0 [ 8.5,  9.5, 10.5, 11.5],
    #  0 [12.5, 13.5, 14.5, 15.5],
    kernel_shape = [2, 2]
    auto_pad = "same_lower"
    rounding_type = "floor"
    index_et = "i32"

    data_node = ov.parameter(data.shape, name="A", dtype=np.float32)
    maxpool_node = ov.max_pool(
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
    comp = rt.computation(maxpool_node, data_node)
    result = comp(data)

    expected = np.array(
        [
            [
                [
                    [0.5, 1.5, 2.5, 3.5],
                    [4.5, 5.5, 6.5, 7.5],
                    [8.5, 9.5, 10.5, 11.5],
                    [12.5, 13.5, 14.5, 15.5],
                ]
            ]
        ],
        dtype=np.float32,
    )
    expected_idx = np.array(
        [
            [
                [
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [8, 9, 10, 11],
                    [12, 13, 14, 15],
                ]
            ]
        ],
        dtype=np.int32,
    )
    assert np.allclose(result[0], expected)
    assert np.allclose(result[1], expected_idx)
