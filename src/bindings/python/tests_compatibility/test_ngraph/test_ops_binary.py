# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Type


@pytest.mark.parametrize(
    ("ng_api_helper", "numpy_function", "expected_type"),
    [
        (ng.add, np.add, Type.f32),
        (ng.divide, np.divide, Type.f32),
        (ng.multiply, np.multiply, Type.f32),
        (ng.subtract, np.subtract, Type.f32),
        (ng.minimum, np.minimum, Type.f32),
        (ng.maximum, np.maximum, Type.f32),
        (ng.mod, np.mod, Type.f32),
        (ng.equal, np.equal, Type.boolean),
        (ng.not_equal, np.not_equal, Type.boolean),
        (ng.greater, np.greater, Type.boolean),
        (ng.greater_equal, np.greater_equal, Type.boolean),
        (ng.less, np.less, Type.boolean),
        (ng.less_equal, np.less_equal, Type.boolean),
    ],
)
def test_binary_op(ng_api_helper, numpy_function, expected_type):
    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)
    parameter_b = ng.parameter(shape, name="B", dtype=np.float32)

    model = ng_api_helper(parameter_a, parameter_b)

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("ng_api_helper", "numpy_function", "expected_type"),
    [
        (ng.add, np.add, Type.f32),
        (ng.divide, np.divide, Type.f32),
        (ng.multiply, np.multiply, Type.f32),
        (ng.subtract, np.subtract, Type.f32),
        (ng.minimum, np.minimum, Type.f32),
        (ng.maximum, np.maximum, Type.f32),
        (ng.mod, np.mod, Type.f32),
        (ng.equal, np.equal, Type.boolean),
        (ng.not_equal, np.not_equal, Type.boolean),
        (ng.greater, np.greater, Type.boolean),
        (ng.greater_equal, np.greater_equal, Type.boolean),
        (ng.less, np.less, Type.boolean),
        (ng.less_equal, np.less_equal, Type.boolean),
    ],
)
def test_binary_op_with_scalar(ng_api_helper, numpy_function, expected_type):
    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = ng_api_helper(parameter_a, value_b)
    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    "ng_api_helper,numpy_function",
    [(ng.logical_and, np.logical_and), (ng.logical_or, np.logical_or), (ng.logical_xor, np.logical_xor)],
)
def test_binary_logical_op(ng_api_helper, numpy_function):
    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.bool)
    parameter_b = ng.parameter(shape, name="B", dtype=np.bool)

    model = ng_api_helper(parameter_a, parameter_b)

    value_a = np.array([[True, False], [False, True]], dtype=np.bool)
    value_b = np.array([[False, True], [False, True]], dtype=np.bool)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == Type.boolean


@pytest.mark.parametrize(
    "ng_api_helper,numpy_function",
    [(ng.logical_and, np.logical_and), (ng.logical_or, np.logical_or), (ng.logical_xor, np.logical_xor)],
)
def test_binary_logical_op_with_scalar(ng_api_helper, numpy_function):
    value_a = np.array([[True, False], [False, True]], dtype=np.bool)
    value_b = np.array([[False, True], [False, True]], dtype=np.bool)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.bool)

    model = ng_api_helper(parameter_a, value_b)
    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)


@pytest.mark.parametrize(
    ("operator", "numpy_function", "expected_type"),
    [
        (operator.add, np.add, Type.f32),
        (operator.sub, np.subtract, Type.f32),
        (operator.mul, np.multiply, Type.f32),
        (operator.truediv, np.divide, Type.f32),
        (operator.eq, np.equal, Type.boolean),
        (operator.ne, np.not_equal, Type.boolean),
        (operator.gt, np.greater, Type.boolean),
        (operator.ge, np.greater_equal, Type.boolean),
        (operator.lt, np.less, Type.boolean),
        (operator.le, np.less_equal, Type.boolean),
    ],
)
def test_binary_operators(operator, numpy_function, expected_type):
    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[4, 5], [1, 7]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)
    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("operator", "numpy_function", "expected_type"),
    [
        (operator.add, np.add, Type.f32),
        (operator.sub, np.subtract, Type.f32),
        (operator.mul, np.multiply, Type.f32),
        (operator.truediv, np.divide, Type.f32),
        (operator.eq, np.equal, Type.boolean),
        (operator.ne, np.not_equal, Type.boolean),
        (operator.gt, np.greater, Type.boolean),
        (operator.ge, np.greater_equal, Type.boolean),
        (operator.lt, np.less, Type.boolean),
        (operator.le, np.less_equal, Type.boolean),
    ],
)
def test_binary_operators_with_scalar(operator, numpy_function, expected_type):
    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)
    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == expected_type


def test_multiply():
    A = np.arange(48, dtype=np.int32).reshape((8, 1, 6, 1))
    B = np.arange(35, dtype=np.int32).reshape((7, 1, 5))

    expected_shape = np.multiply(A, B).shape
    node = ng.multiply(A, B)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(expected_shape)
    assert node.get_output_element_type(0) == Type.i32


def test_power_v1():
    A = np.arange(48, dtype=np.float32).reshape((8, 1, 6, 1))
    B = np.arange(20, dtype=np.float32).reshape((4, 1, 5))

    expected_shape = np.power(A, B).shape
    node = ng.power(A, B)

    assert node.get_type_name() == "Power"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(expected_shape)
    assert node.get_output_element_type(0) == Type.f32
