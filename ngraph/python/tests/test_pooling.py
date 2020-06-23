# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import pytest

import ngraph as ng
from tests.util import get_runtime


@pytest.fixture
def _ndarray_1x1x4x4():
    return np.arange(11, 27, dtype=np.float32).reshape(1, 1, 4, 4)


def test_avg_pool_2d(_ndarray_1x1x4x4):
    runtime = get_runtime()
    input_data = _ndarray_1x1x4x4
    param = ng.parameter(input_data.shape, name="A", dtype=np.float32)

    kernel_shape = [2, 2]
    spatial_dim_count = len(kernel_shape)
    pads_begin = [0] * spatial_dim_count
    pads_end = [0] * spatial_dim_count
    strides = [2, 2]
    exclude_pad = True
    expected = [[[[13.5, 15.5], [21.5, 23.5]]]]

    avg_pool_node = ng.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    expected = [[[[13.5, 14.5, 15.5], [17.5, 18.5, 19.5], [21.5, 22.5, 23.5]]]]
    strides = [1, 1]
    avg_pool_node = ng.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    pads_begin = [1, 1]
    pads_end = [1, 1]
    strides = [2, 2]
    exclude_pad = True

    expected = [[[[11.0, 12.5, 14.0], [17.0, 18.5, 20.0], [23.0, 24.5, 26.0]]]]
    avg_pool_node = ng.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    exclude_pad = False
    expected = [[[[2.75, 6.25, 3.5], [8.5, 18.5, 10.0], [5.75, 12.25, 6.5]]]]
    avg_pool_node = ng.avg_pool(param, strides, pads_begin, pads_end, kernel_shape, exclude_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)


def test_avg_pooling_3d(_ndarray_1x1x4x4):
    rt = get_runtime()
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
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [2, 2]

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    avgpool_node = ng.max_pool(data_node, strides, pads_begin, pads_end, kernel_shape)
    comp = rt.computation(avgpool_node, data_node)
    result = comp(data)

    expected = np.array([[[[5.5, 6.5, 7.5], [9.5, 10.5, 11.5], [13.5, 14.5, 15.5]]]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_max_pool_strides():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [2, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [2, 2]

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    avgpool_node = ng.max_pool(data_node, strides, pads_begin, pads_end, kernel_shape)
    comp = rt.computation(avgpool_node, data_node)
    result = comp(data)

    expected = np.array([[[[5.5, 6.5, 7.5], [13.5, 14.5, 15.5]]]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_max_pool_kernel_shape1d():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [1, 1]

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    avgpool_node = ng.max_pool(data_node, strides, pads_begin, pads_end, kernel_shape)
    comp = rt.computation(avgpool_node, data_node)
    result = comp(data)

    assert np.allclose(result, data)


def test_max_pool_kernel_shape3d():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    kernel_shape = [3, 3]

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    avgpool_node = ng.max_pool(data_node, strides, pads_begin, pads_end, kernel_shape)
    comp = rt.computation(avgpool_node, data_node)
    result = comp(data)

    expected = np.array([[[[10.5, 11.5], [14.5, 15.5]]]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_max_pool_non_zero_pads():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    pads_begin = [1, 1]
    pads_end = [1, 1]
    #  0   0  ,  0  ,  0  ,  0,    0
    #  0 [ 0.5,  1.5,  2.5,  3.5], 0,
    #  0 [ 4.5,  5.5,  6.5,  7.5], 0,
    #  0 [ 8.5,  9.5, 10.5, 11.5], 0,
    #  0 [12.5, 13.5, 14.5, 15.5], 0
    #  0   0  ,  0  ,  0  ,  0,    0
    kernel_shape = [2, 2]

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    avgpool_node = ng.max_pool(data_node, strides, pads_begin, pads_end, kernel_shape)
    comp = rt.computation(avgpool_node, data_node)
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
    assert np.allclose(result, expected)


def test_max_pool_same_upper_auto_pads():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    # [ 0.5,  1.5,  2.5,  3.5], 0,
    # [ 4.5,  5.5,  6.5,  7.5], 0,
    # [ 8.5,  9.5, 10.5, 11.5], 0,
    # [12.5, 13.5, 14.5, 15.5], 0
    #   0  ,  0  ,  0  ,  0,    0
    kernel_shape = [2, 2]
    auto_pad = "same_upper"

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    avgpool_node = ng.max_pool(data_node, strides, pads_begin, pads_end, kernel_shape, auto_pad=auto_pad)
    comp = rt.computation(avgpool_node, data_node)
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
    assert np.allclose(result, expected)


def test_max_pool_same_lower_auto_pads():
    rt = get_runtime()

    # array([[[[ 0.5,  1.5,  2.5,  3.5],
    #          [ 4.5,  5.5,  6.5,  7.5],
    #          [ 8.5,  9.5, 10.5, 11.5],
    #          [12.5, 13.5, 14.5, 15.5]]]], dtype=float32)
    data = np.arange(0.5, 16, dtype=np.float32).reshape((1, 1, 4, 4))
    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    #  0   0  ,  0  ,  0  ,  0,
    #  0 [ 0.5,  1.5,  2.5,  3.5],
    #  0 [ 4.5,  5.5,  6.5,  7.5],
    #  0 [ 8.5,  9.5, 10.5, 11.5],
    #  0 [12.5, 13.5, 14.5, 15.5],
    kernel_shape = [2, 2]
    auto_pad = "same_lower"

    data_node = ng.parameter(data.shape, name="A", dtype=np.float32)
    avgpool_node = ng.max_pool(data_node, strides, pads_begin, pads_end, kernel_shape, auto_pad=auto_pad)
    comp = rt.computation(avgpool_node, data_node)
    result = comp(data)

    expected = np.array(
        [[[[0.5, 1.5, 2.5, 3.5], [4.5, 5.5, 6.5, 7.5], [8.5, 9.5, 10.5, 11.5], [12.5, 13.5, 14.5, 15.5],]]],
        dtype=np.float32,
    )
    assert np.allclose(result, expected)
