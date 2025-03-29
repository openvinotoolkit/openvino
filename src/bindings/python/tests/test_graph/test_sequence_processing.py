# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino as ov


@pytest.mark.parametrize(("depth", "on_value", "off_value", "axis", "expected_shape"), [
    (2, 5, 10, -1, [3, 2]),
    (3, 1, 0, 0, [3, 3]),
])
def test_one_hot(depth, on_value, off_value, axis, expected_shape):
    param = ov.opset11.parameter([3], dtype=np.int32)
    node = ov.opset11.one_hot(param, depth, on_value, off_value, axis)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "OneHot"
    assert list(node.get_output_shape(0)) == expected_shape


# Test Range-1
def test_range_1():
    start = 5
    stop = 35
    step = 5

    node = ov.opset11.range(start, stop, step)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Range"
    assert list(node.get_output_shape(0)) == [6]


# Test Range-4
@pytest.mark.parametrize(("destination_type", "expected_type"), [
    ("i64", ov.Type.i64),
    ("i32", ov.Type.i32),
    ("f32", ov.Type.f32),
])
def test_range_4(destination_type, expected_type):
    start = 5
    stop = 35
    step = 5

    node = ov.opset12.range(start, stop, step, destination_type)

    assert node.get_output_size() == 1
    assert node.get_type_name() == "Range"
    assert list(node.get_output_shape(0)) == [6]
    assert node.get_element_type() == expected_type
