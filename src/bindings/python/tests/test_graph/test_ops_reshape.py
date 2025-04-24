# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.opset8 as ov
import numpy as np
import pytest

from openvino import Type
from openvino.utils.types import get_element_type


@pytest.mark.parametrize("op_name", ["ABC", "concat", "123456"])
def test_concat(op_name):
    input_a = np.array([[1, 2], [3, 4]]).astype(np.float32)
    input_b = np.array([[5, 6]]).astype(np.float32)
    axis = 0

    parameter_a = ov.parameter(list(input_a.shape), name="A", dtype=np.float32)
    parameter_b = ov.parameter(list(input_b.shape), name="B", dtype=np.float32)
    node = ov.concat([parameter_a, parameter_b], axis, name=op_name)
    assert node.get_type_name() == "Concat"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]


@pytest.mark.parametrize(
    ("val_type", "value", "output_shape"), [(bool, False, []), (bool, np.empty((2, 2), dtype=bool), [2, 2])],
)
def test_constant_from_bool(val_type, value, output_shape):
    node = ov.constant(value, val_type)
    assert node.get_type_name() == "Constant"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.boolean
    assert list(node.get_output_shape(0)) == output_shape


@pytest.mark.parametrize(
    ("val_type", "value"),
    [
        pytest.param(np.int16, np.int16(-12345)),
        pytest.param(np.int64, np.int64(-1234567)),
        pytest.param(np.uint16, np.uint16(12345)),
        pytest.param(np.uint32, np.uint32(123456)),
        pytest.param(np.uint64, np.uint64(1234567)),
        pytest.param(np.float64, np.float64(0.1234)),
        pytest.param(np.float32, np.float32(0.1234)),
        pytest.param(np.int8, np.int8(-63)),
        pytest.param(np.int32, np.int32(-123456)),
        pytest.param(np.uint8, np.uint8(63)),
    ],
)
def test_constant_from_scalar(val_type, value):
    node = ov.constant(value, val_type)
    assert node.get_type_name() == "Constant"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(val_type)
    assert list(node.get_output_shape(0)) == []


@pytest.mark.parametrize(
    "val_type",
    [
        pytest.param(np.float64),
        pytest.param(np.float32),
    ],
)
def test_constant_from_float_array(val_type):
    np.random.seed(133391)
    input_data = np.array(-1 + np.random.rand(2, 3, 4) * 2, dtype=val_type)
    node = ov.constant(input_data, val_type)
    assert node.get_type_name() == "Constant"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(val_type)
    assert list(node.get_output_shape(0)) == [2, 3, 4]


@pytest.mark.parametrize(
    ("val_type", "range_start", "range_end"),
    [
        pytest.param(np.int16, -64, 64),
        pytest.param(np.int64, -16383, 16383),
        pytest.param(np.uint16, 0, 64),
        pytest.param(np.uint32, 0, 1024),
        pytest.param(np.uint64, 0, 16383),
        pytest.param(np.int8, -8, 8),
        pytest.param(np.int32, -1024, 1024),
        pytest.param(np.uint8, 0, 8),
    ],
)
def test_constant_from_integer_array(val_type, range_start, range_end):
    np.random.seed(133391)
    input_data = np.array(
        np.random.randint(range_start, range_end, size=(2, 2)), dtype=val_type,
    )
    node = ov.constant(input_data, val_type)
    assert node.get_type_name() == "Constant"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == get_element_type(val_type)
    assert list(node.get_output_shape(0)) == [2, 2]


@pytest.mark.parametrize("op_name", ["broadcast", "123412"])
def test_broadcast_numpy(op_name):
    data_shape = [16, 1, 1]
    target_shape_shape = [4]

    data_parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)
    target_shape_parameter = ov.parameter(
        target_shape_shape, name="Target_shape", dtype=np.int64,
    )

    node = ov.broadcast(data_parameter, target_shape_parameter, name=op_name)

    assert node.get_type_name() == "Broadcast"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1


@pytest.mark.parametrize("op_name", ["broadcast", "broadcast_bidiretional"])
def test_broadcast_bidirectional(op_name):
    data_shape = [16, 1, 1]
    target_shape_shape = [4]

    data_parameter = ov.parameter(data_shape, name="Data", dtype=np.float32)
    target_shape_parameter = ov.parameter(
        target_shape_shape, name="Target_shape", dtype=np.int64,
    )

    node = ov.broadcast(data_parameter, target_shape_parameter, name=op_name)

    assert node.get_type_name() == "Broadcast"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1


@pytest.mark.parametrize("op_name", ["transpose", "transpose123"])
def test_transpose(op_name):
    input_tensor = np.arange(3 * 3 * 224 * 224, dtype=np.int32).reshape(
        (3, 3, 224, 224),
    )
    input_order = np.array([0, 2, 3, 1], dtype=np.int32)

    node = ov.transpose(input_tensor, input_order, name=op_name)
    assert node.get_type_name() == "Transpose"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.i32
    assert list(node.get_output_shape(0)) == [3, 224, 224, 3]


@pytest.mark.parametrize("op_name", ["tile", "tile123"])
def test_tile(op_name):
    input_tensor = np.arange(6, dtype=np.int32).reshape((2, 1, 3))
    repeats = np.array([2, 1], dtype=np.int32)
    node = ov.tile(input_tensor, repeats, name=op_name)

    assert node.get_type_name() == "Tile"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.i32
    assert list(node.get_output_shape(0)) == [2, 2, 3]


@pytest.mark.parametrize("op_name", ["slice", "strided_slice"])
def test_strided_slice(op_name):
    input_tensor = np.arange(2 * 3 * 4, dtype=np.float32).reshape((2, 3, 4))
    begin = np.array([1, 0], dtype=np.int32)
    end = np.array([0, 0], dtype=np.int32)
    strides = np.array([1, 1], dtype=np.int32)
    begin_mask = np.array([0, 0, 0], dtype=np.int32)
    end_mask = np.array([0, 0, 0], dtype=np.int32)
    new_axis_mask = np.array([0, 1, 0], dtype=np.int32)
    shrink_axis_mask = np.array([1, 0, 0], dtype=np.int32)
    ellipsis_mask = np.array([0, 0, 0], dtype=np.int32)

    node = ov.strided_slice(
        input_tensor,
        begin,
        end,
        strides,
        begin_mask,
        end_mask,
        new_axis_mask,
        shrink_axis_mask,
        ellipsis_mask,
        name=op_name,
    )
    assert node.get_type_name() == "StridedSlice"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [1, 3, 4]


@pytest.mark.parametrize("op_name", ["reshape", "reshapev1"])
def test_reshape_v1(op_name):
    param_a = np.arange(1200, dtype=np.float32).reshape((2, 5, 5, 24))
    shape = np.array([0, -1, 4], dtype=np.int32)
    special_zero = True

    node = ov.reshape(param_a, shape, special_zero, name=op_name)
    assert node.get_type_name() == "Reshape"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [2, 150, 4]


@pytest.mark.parametrize("op_name", ["shape", "shape_of"])
def test_shape_of(op_name):
    input_tensor = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    node = ov.shape_of(input_tensor, name=op_name)
    assert node.get_type_name() == "ShapeOf"
    assert node.get_friendly_name() == op_name
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.i64
    assert list(node.get_output_shape(0)) == [2]
