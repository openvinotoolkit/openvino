# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset14 as ov
from openvino import Type


avg_pooling_test_params = [
    (
        [
            [2, 2],  # strides
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [2, 2],  # kernel_shape
            True,  # exclude_pad
            "floor",  # rounding_type
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 2, 2],  # expected_output_shape
    ),
    (
        [
            [2, 2],  # strides
            [1, 1],  # pads_begin
            [1, 1],  # pads_end
            [2, 2],  # kernel_shape
            False,  # exclude_pad
            "ceil_torch",  # rounding_type
        ],
        [1, 1, 5, 5],  # input_shape
        [1, 1, 3, 3],  # expected_output_shape
    ),
    (
        [
            [2, 2],  # strides
            [1, 1],  # pads_begin
            [1, 1],  # pads_end
            [2, 2],  # kernel_shape
            False,  # exclude_pad
            "ceil_torch",  # rounding_type
        ],
        [1, 3, 9, 9],  # input_shape
        [1, 3, 5, 5],  # expected_output_shape
    ),
    (
        [
            [2, 2],  # strides
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [3, 3],  # kernel_shape
            False,  # exclude_pad
            "ceil_torch",  # rounding_type
        ],
        [1, 3, 10, 10],  # input_shape
        [1, 3, 5, 5],  # expected_output_shape
    ),
    (
        [
            [1, 1],  # strides
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [3, 3],  # kernel_shape
            False,  # exclude_pad
            "ceil_torch",  # rounding_type
        ],
        [1, 3, 10, 10],  # input_shape
        [1, 3, 8, 8],  # expected_output_shape
    ),
    (
        [
            [2, 2, 2],  # strides
            [0, 0, 0],  # pads_begin
            [0, 0, 0],  # pads_end
            [2, 2, 2],  # kernel_shape
            True,  # exclude_pad
            "ceil_torch",  # rounding_type
        ],
        [1, 1, 4, 4, 4],  # input_shape
        [1, 1, 2, 2, 2],  # expected_output_shape
    ),
]


max_pooling_test_params = [
    (
        [
            [1, 1],  # strides
            [1, 1],  # dilations
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [2, 2],  # kernel_shape
            "floor",  # rounding_type
            None,  # auto_pad
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 3, 3],  # expected_output_shape
    ),
    (
        [
            [2, 1],  # strides
            [1, 1],  # dilations
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [2, 2],  # kernel_shape
            "floor",  # rounding_type
            None,  # auto_pad
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 2, 3],  # expected_output_shape
    ),
    (
        [
            [1, 1],  # strides
            [1, 1],  # dilations
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [1, 1],  # kernel_shape
            "floor",  # rounding_type
            None,  # auto_pad
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 4, 4],  # expected_output_shape
    ),
    (
        [
            [1, 1],  # strides
            [1, 1],  # dilations
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [3, 3],  # kernel_shape
            "floor",  # rounding_type
            None,  # auto_pad
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 2, 2],  # expected_output_shape
    ),
    (
        [
            [1, 1],  # strides
            [1, 1],  # dilations
            [1, 1],  # pads_begin
            [1, 1],  # pads_end
            [2, 2],  # kernel_shape
            "floor",  # rounding_type
            None,  # auto_pad
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 5, 5],  # expected_output_shape
    ),
    (
        [
            [1, 1],  # strides
            [1, 1],  # dilations
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [2, 2],  # kernel_shape
            "floor",  # rounding_type
            "same_upper",  # auto_pad
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 4, 4],  # expected_output_shape
    ),
    (
        [
            [2, 2],  # strides
            [1, 1],  # dilations
            [1, 1],  # pads_begin
            [1, 1],  # pads_end
            [2, 2],  # kernel_shape
            "ceil_torch",  # rounding_type
            None,  # auto_pad
        ],
        [1, 1, 5, 5],  # input_shape
        [1, 1, 3, 3],  # expected_output_shape
    ),
    (
        [
            [1, 1],  # strides
            [1, 1],  # dilations
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [2, 2],  # kernel_shape
            "ceil_torch",  # rounding_type
            "same_lower",  # auto_pad
        ],
        [1, 1, 4, 4],  # input_shape
        [1, 1, 4, 4],  # expected_output_shape
    ),
    (
        [
            [1, 1],  # strides
            [1, 1],  # dilations
            [0, 0],  # pads_begin
            [0, 0],  # pads_end
            [3, 3],  # kernel_shape
            "ceil_torch",  # rounding_type
            None,  # auto_pad
        ],
        [1, 1, 10, 10],  # input_shape
        [1, 1, 8, 8],  # expected_output_shape
    ),
]


@pytest.mark.parametrize(
    ("op_params", "input_shape", "expected_output_shape"),
    avg_pooling_test_params,
)
@pytest.mark.parametrize("op_name", ["avg_pool", "avgPool", "avgPoolOpset14"])
def test_avg_pool(op_params, input_shape, expected_output_shape, op_name):
    param = ov.parameter(input_shape, name="A", dtype=np.float32)
    node = ov.avg_pool(param, *op_params, name=op_name)
    assert node.get_type_name() == "AvgPool"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_output_shape
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(
    ("op_params", "input_shape", "expected_output_shape"),
    max_pooling_test_params,
)
@pytest.mark.parametrize("op_name", ["avg_pool", "avgPool", "avgPoolOpset14"])
def test_max_pool(op_params, input_shape, expected_output_shape, op_name):
    data_node = ov.parameter(input_shape, name="A", dtype=np.float32)
    node = ov.max_pool(data_node, *op_params, "i32", name=op_name)
    assert node.get_type_name() == "MaxPool"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == expected_output_shape
    assert list(node.get_output_shape(1)) == expected_output_shape
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.i32
