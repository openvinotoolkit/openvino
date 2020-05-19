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

import ngraph as ng
from test.ngraph.util import get_runtime, run_op_node
from test.test_ops import convolution2d


def test_convolution_2d():

    # input_x should have shape N(batch) x C x H x W
    input_x = np.array(
        [
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    ).reshape(1, 1, 9, 9)

    # filter weights should have shape M x C x kH x kW
    input_filter = np.array(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32
    ).reshape(1, 1, 3, 3)

    strides = np.array([1, 1])
    pads_begin = np.array([1, 1])
    pads_end = np.array([1, 1])
    dilations = np.array([1, 1])

    # convolution with padding=1 should produce 9 x 9 output:
    result = run_op_node(
        [input_x, input_filter], ng.convolution, strides, pads_begin, pads_end, dilations
    )

    assert np.allclose(
        result,
        np.array(
            [
                [
                    [
                        [0.0, -15.0, -15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -20.0, -20.0, 20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -15.0, -15.0, 15.0, 15.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )

    # convolution with padding=0 should produce 7 x 7 output:
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])
    result = run_op_node(
        [input_x, input_filter], ng.convolution, strides, pads_begin, pads_end, dilations
    )
    assert np.allclose(
        result,
        np.array(
            [
                [
                    [
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                        [-20, -20, 20, 20, 0, 0, 0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )

    strides = np.array([2, 2])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])

    # convolution with strides=2 should produce 4 x 4 output:
    result = run_op_node(
        [input_x, input_filter], ng.convolution, strides, pads_begin, pads_end, dilations
    )

    assert np.allclose(
        result,
        np.array(
            [
                [
                    [
                        [-20.0, 20.0, 0.0, 0.0],
                        [-20.0, 20.0, 0.0, 0.0],
                        [-20.0, 20.0, 0.0, 0.0],
                        [-20.0, 20.0, 0.0, 0.0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )

    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([2, 2])

    # convolution with dilation=2 should produce 5 x 5 output:
    result = run_op_node(
        [input_x, input_filter], ng.convolution, strides, pads_begin, pads_end, dilations
    )
    assert np.allclose(
        result,
        np.array(
            [
                [
                    [
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                        [0, 0, 20, 20, 0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )


def test_convolution_backprop_data():
    runtime = get_runtime()

    output_spatial_shape = [9, 9]
    filter_shape = [1, 1, 3, 3]
    data_shape = [1, 1, 7, 7]
    strides = [1, 1]

    data_node = ng.parameter(shape=data_shape)
    filter_node = ng.parameter(shape=filter_shape)
    output_shape_node = ng.constant(np.array(output_spatial_shape, dtype=np.int64))

    deconvolution = ng.convolution_backprop_data(data_node, filter_node, strides, output_shape_node)

    input_data = np.array(
        [
            [
                [
                    [-20, -20, 20, 20, 0, 0, 0],
                    [-20, -20, 20, 20, 0, 0, 0],
                    [-20, -20, 20, 20, 0, 0, 0],
                    [-20, -20, 20, 20, 0, 0, 0],
                    [-20, -20, 20, 20, 0, 0, 0],
                    [-20, -20, 20, 20, 0, 0, 0],
                    [-20, -20, 20, 20, 0, 0, 0],
                ]
            ]
        ],
        dtype=np.float32,
    )

    filter_data = np.array(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=np.float32
    ).reshape(1, 1, 3, 3)

    model = runtime.computation(deconvolution, data_node, filter_node)
    result = model(input_data, filter_data)
    assert np.allclose(
        result,
        np.array(
            [
                [
                    [
                        [-20.0, -20.0, 40.0, 40.0, -20.0, -20.0, 0.0, 0.0, 0.0],
                        [-60.0, -60.0, 120.0, 120.0, -60.0, -60.0, 0.0, 0.0, 0.0],
                        [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                        [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                        [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                        [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                        [-80.0, -80.0, 160.0, 160.0, -80.0, -80.0, 0.0, 0.0, 0.0],
                        [-60.0, -60.0, 120.0, 120.0, -60.0, -60.0, 0.0, 0.0, 0.0],
                        [-20.0, -20.0, 40.0, 40.0, -20.0, -20.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            dtype=np.float32,
        ),
    )


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

    result = run_op_node(
        [input_tensor, filters], ng.convolution, strides, pads_begin, pads_end, dilations
    )

    expected = convolution2d(input_tensor[0, 0], filters[0, 0]).reshape(1, 1, 14, 14)

    assert np.allclose(result, expected)
