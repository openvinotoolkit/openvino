# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pytest

from openvino.runtime import Type
import openvino.runtime.opset8 as ov


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function", "expected_type"),
    [
        (ov.add, np.add, Type.f32),
        (ov.divide, np.divide, Type.f32),
        (ov.multiply, np.multiply, Type.f32),
        (ov.subtract, np.subtract, Type.f32),
        (ov.minimum, np.minimum, Type.f32),
        (ov.maximum, np.maximum, Type.f32),
        (ov.mod, np.mod, Type.f32),
        (ov.equal, np.equal, Type.boolean),
        (ov.not_equal, np.not_equal, Type.boolean),
        (ov.greater, np.greater, Type.boolean),
        (ov.greater_equal, np.greater_equal, Type.boolean),
        (ov.less, np.less, Type.boolean),
        (ov.less_equal, np.less_equal, Type.boolean),
    ],
)
def test_binary_op(graph_api_helper, numpy_function, expected_type):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)
    parameter_b = ov.parameter(shape, name="B", dtype=np.float32)

    model = graph_api_helper(parameter_a, parameter_b)

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function", "expected_type"),
    [
        (ov.add, np.add, Type.f32),
        (ov.divide, np.divide, Type.f32),
        (ov.multiply, np.multiply, Type.f32),
        (ov.subtract, np.subtract, Type.f32),
        (ov.minimum, np.minimum, Type.f32),
        (ov.maximum, np.maximum, Type.f32),
        (ov.mod, np.mod, Type.f32),
        (ov.equal, np.equal, Type.boolean),
        (ov.not_equal, np.not_equal, Type.boolean),
        (ov.greater, np.greater, Type.boolean),
        (ov.greater_equal, np.greater_equal, Type.boolean),
        (ov.less, np.less, Type.boolean),
        (ov.less_equal, np.less_equal, Type.boolean),
    ],
)
def test_binary_op_with_scalar(graph_api_helper, numpy_function, expected_type):
    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = graph_api_helper(parameter_a, value_b)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function"),
    [(ov.logical_and, np.logical_and), (ov.logical_or, np.logical_or), (ov.logical_xor, np.logical_xor)],
)
def test_binary_logical_op(graph_api_helper, numpy_function):
    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)
    parameter_b = ov.parameter(shape, name="B", dtype=bool)

    model = graph_api_helper(parameter_a, parameter_b)

    value_a = np.array([[True, False], [False, True]], dtype=bool)
    value_b = np.array([[False, True], [False, True]], dtype=bool)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == Type.boolean


@pytest.mark.parametrize(
    ("graph_api_helper", "numpy_function"),
    [(ov.logical_and, np.logical_and), (ov.logical_or, np.logical_or), (ov.logical_xor, np.logical_xor)],
)
def test_binary_logical_op_with_scalar(graph_api_helper, numpy_function):
    value_a = np.array([[True, False], [False, True]], dtype=bool)
    value_b = np.array([[False, True], [False, True]], dtype=bool)

    shape = [2, 2]
    parameter_a = ov.parameter(shape, name="A", dtype=bool)

    model = graph_api_helper(parameter_a, value_b)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == Type.boolean


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
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

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
    parameter_a = ov.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)

    expected_shape = numpy_function(value_a, value_b).shape
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == list(expected_shape)
    assert model.get_output_element_type(0) == expected_type


def test_multiply():
    param_a = np.arange(48, dtype=np.int32).reshape((8, 1, 6, 1))
    param_b = np.arange(35, dtype=np.int32).reshape((7, 1, 5))

    expected_shape = np.multiply(param_a, param_b).shape
    node = ov.multiply(param_a, param_b)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(expected_shape)
    assert node.get_output_element_type(0) == Type.i32


def test_power_v1():
    param_a = np.arange(48, dtype=np.float32).reshape((8, 1, 6, 1))
    param_b = np.arange(20, dtype=np.float32).reshape((4, 1, 5))

    expected_shape = np.power(param_a, param_b).shape
    node = ov.power(param_a, param_b)

    assert node.get_type_name() == "Power"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(expected_shape)
    assert node.get_output_element_type(0) == Type.f32
