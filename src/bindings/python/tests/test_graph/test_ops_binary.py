# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pytest
import warnings

from openvino import Type
import openvino.opset13 as ov
import openvino.opset15 as ov_opset15


@pytest.mark.parametrize(
    ("graph_api_helper", "expected_type"),
    [
        (ov.add, Type.f32),
        (ov.divide, Type.f32),
        (ov.multiply, Type.f32),
        (ov.subtract, Type.f32),
        (ov.minimum, Type.f32),
        (ov.maximum, Type.f32),
        (ov.mod, Type.f32),
        (ov.equal, Type.boolean),
        (ov.not_equal, Type.boolean),
        (ov.greater, Type.boolean),
        (ov.greater_equal, Type.boolean),
        (ov.less, Type.boolean),
        (ov.less_equal, Type.boolean),
    ],
)
def test_binary_op(graph_api_helper, expected_type):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)
    parameter_b = ov.parameter(shape, name="B", dtype=np.float32)

    model = graph_api_helper(parameter_a, parameter_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("graph_api_helper", "expected_type"),
    [
        (ov.add, Type.f32),
        (ov.divide, Type.f32),
        (ov.multiply, Type.f32),
        (ov.subtract, Type.f32),
        (ov.minimum, Type.f32),
        (ov.maximum, Type.f32),
        (ov.mod, Type.f32),
        (ov.equal, Type.boolean),
        (ov.not_equal, Type.boolean),
        (ov.greater, Type.boolean),
        (ov.greater_equal, Type.boolean),
        (ov.less, Type.boolean),
        (ov.less_equal, Type.boolean),
    ],
)
def test_binary_op_with_scalar(graph_api_helper, expected_type):
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = graph_api_helper(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    "graph_api_helper",
    [ov.logical_and, ov.logical_or, ov.logical_xor],
)
def test_binary_logical_op(graph_api_helper):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)
    parameter_b = ov.parameter(shape, name="B", dtype=bool)

    model = graph_api_helper(parameter_a, parameter_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == Type.boolean


@pytest.mark.parametrize(
    "graph_api_helper",
    [ov.logical_and, ov.logical_or, ov.logical_xor],
)
def test_binary_logical_op_with_scalar(graph_api_helper):
    value_b = np.array([[False, True], [False, True]], dtype=bool)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)

    model = graph_api_helper(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == Type.boolean


@pytest.mark.parametrize(
    ("operator", "expected_ov_str", "expected_type"),
    [
        (operator.add, "Add", Type.f32),
        (operator.sub, "Subtract", Type.f32),
        (operator.mul, "Multiply", Type.f32),
        (operator.truediv, "Divide", Type.f32),
    ],
)
@pytest.mark.parametrize(
    "value_b",
    [
        np.array([[4, 5], [1, 7]], dtype=np.float32),
        np.array([[5, 6], [7, 8]], dtype=np.float32),
        ov.parameter([2, 2], name="B", dtype=np.float32),
    ],
)
def test_binary_operators(operator, expected_ov_str, expected_type, value_b):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)

    assert model.get_type_name() == expected_ov_str
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("operator", "expected_ov_str"),
    [
        (operator.add, "Add"),
        (operator.sub, "Subtract"),
        (operator.mul, "Multiply"),
        (operator.truediv, "Divide"),
    ],
)
@pytest.mark.parametrize(
    ("expected_type", "value_a", "value_b"),
    [
        (Type.f64, ov.parameter([2, 2], name="A", dtype=np.float64), 3.12),
        (Type.i64, ov.parameter([2, 2], name="A", dtype=np.int64), 2),
        (Type.f64, 3.12, ov.parameter([2, 2], name="A", dtype=np.float64)),
        (Type.i64, 2, ov.parameter([2, 2], name="A", dtype=np.int64)),
        (Type.f32, np.array([[4, 5], [1, 7]], dtype=np.float32), ov.parameter([2, 2], name="B", dtype=np.float32)),
    ],
)
def test_binary_operators_rside(operator, expected_ov_str, expected_type, value_a, value_b):
    shape = [2, 2]
    model = operator(value_a, value_b)

    assert model.get_type_name() == expected_ov_str
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == expected_type


def test_multiply():
    param_a = np.arange(48, dtype=np.int32).reshape((8, 1, 6, 1))
    param_b = np.arange(35, dtype=np.int32).reshape((7, 1, 5))

    node = ov.multiply(param_a, param_b)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [8, 7, 6, 5]
    assert node.get_output_element_type(0) == Type.i32


def test_power_v1():
    param_a = np.arange(48, dtype=np.float32).reshape((8, 1, 6, 1))
    param_b = np.arange(20, dtype=np.float32).reshape((4, 1, 5))

    node = ov.power(param_a, param_b)

    assert node.get_type_name() == "Power"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [8, 4, 6, 5]
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(
    "graph_api_helper",
    [ov.bitwise_and, ov.bitwise_or, ov.bitwise_xor],
)
@pytest.mark.parametrize(
    "dtype",
    [bool, np.int32],
)
@pytest.mark.parametrize(
    ("shape_a", "shape_b", "broadcast", "shape_out"),
    [
        ([2, 2], [2, 2], "NONE", [2, 2]),
        ([2, 1, 5], [1, 4, 5], "NUMPY", [2, 4, 5]),
        ([3, 2, 1, 4], [5, 4], "NUMPY", [3, 2, 5, 4]),
        ([2, 3, 4, 5], [], "PDPD", [2, 3, 4, 5]),
        ([2, 3, 4, 5], [2, 3, 1, 5], "PDPD", [2, 3, 4, 5]),
    ],
)
def test_binary_bitwise_op(graph_api_helper, dtype, shape_a, shape_b, broadcast, shape_out):
    parameter_a = ov.parameter(shape_a, name="A", dtype=dtype)
    parameter_b = ov.parameter(shape_b, name="B", dtype=dtype)

    model = graph_api_helper(parameter_a, parameter_b, broadcast)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape_out
    assert model.get_output_element_type(0) == Type(dtype)


@pytest.mark.parametrize(
    "graph_api_helper",
    [ov.bitwise_and, ov.bitwise_or, ov.bitwise_xor],
)
@pytest.mark.parametrize(
    "dtype",
    [bool, np.int32],
)
def test_binary_bitwise_op_with_constant(graph_api_helper, dtype):
    value_b = np.array([[3, 0], [-7, 21]], dtype=dtype)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=dtype)

    model = graph_api_helper(parameter_a, value_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape
    assert model.get_output_element_type(0) == Type(dtype)


@pytest.mark.parametrize(
    "graph_api_helper",
    [ov_opset15.bitwise_left_shift, ov_opset15.bitwise_right_shift],
)
@pytest.mark.parametrize(
    "dtype",
    [np.int32],
)
@pytest.mark.parametrize(
    ("shape_a", "shape_b", "broadcast", "shape_out", "is_const"),
    [
        ([2, 2], [2, 2], "NONE", [2, 2], False),
        ([2, 2], [2, 2], "NONE", [2, 2], True),
        ([2, 1, 5], [1, 4, 5], "NUMPY", [2, 4, 5], False),
        ([3, 2, 1, 4], [5, 4], "NUMPY", [3, 2, 5, 4], False),
        ([2, 3, 4, 5], [], "PDPD", [2, 3, 4, 5], False),
        ([2, 3, 4, 5], [2, 3, 1, 5], "PDPD", [2, 3, 4, 5], False),
    ],
)
def test_binary_bitwise_shift_op(graph_api_helper, dtype, shape_a,
                                 shape_b, broadcast, shape_out, is_const):
    parameter_a = ov.parameter(shape_a, name="A", dtype=dtype)
    parameter_b = ov.parameter(shape_b, name="B", dtype=dtype)

    if is_const:
        value_b = np.array([[3, 0], [7, 21]], dtype=dtype)
        model = graph_api_helper(parameter_a, value_b, broadcast)
    else:
        model = graph_api_helper(parameter_a, parameter_b, broadcast)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == shape_out
    assert model.get_output_element_type(0) == Type(dtype)
