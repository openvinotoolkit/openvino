# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import operator

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Type


@pytest.mark.parametrize(
    ("ng_api_helper", "expected_type"),
    [
        (ng.add, Type.f32),
        (ng.divide, Type.f32),
        (ng.multiply, Type.f32),
        (ng.subtract, Type.f32),
        (ng.minimum, Type.f32),
        (ng.maximum, Type.f32),
        (ng.mod, Type.f32),
        (ng.equal, Type.boolean),
        (ng.not_equal, Type.boolean),
        (ng.greater, Type.boolean),
        (ng.greater_equal, Type.boolean),
        (ng.less, Type.boolean),
        (ng.less_equal, Type.boolean),
    ],
)
def test_binary_op(ng_api_helper, expected_type):
    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)
    parameter_b = ng.parameter(shape, name="B", dtype=np.float32)

    model = ng_api_helper(parameter_a, parameter_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("ng_api_helper", "expected_type"),
    [
        (ng.add, Type.f32),
        (ng.divide, Type.f32),
        (ng.multiply, Type.f32),
        (ng.subtract, Type.f32),
        (ng.minimum, Type.f32),
        (ng.maximum, Type.f32),
        (ng.mod, Type.f32),
        (ng.equal, Type.boolean),
        (ng.not_equal, Type.boolean),
        (ng.greater, Type.boolean),
        (ng.greater_equal, Type.boolean),
        (ng.less, Type.boolean),
        (ng.less_equal, Type.boolean),
    ],
)
def test_binary_op(ng_api_helper, expected_type):
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = ng_api_helper(parameter_a, value_b)
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    "ng_api_helper",
    [ng.logical_and, ng.logical_or, ng.logical_xor],
)
def test_binary_logical_op(ng_api_helper):
    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.bool)
    parameter_b = ng.parameter(shape, name="B", dtype=np.bool)

    model = ng_api_helper(parameter_a, parameter_b)

    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == Type.boolean


@pytest.mark.parametrize(
    "ng_api_helper",
    [ng.logical_and, ng.logical_or, ng.logical_xor],
)
def test_binary_logical_op_with_scalar(ng_api_helper):
    value_b = np.array([[False, True], [False, True]], dtype=np.bool)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.bool)

    model = ng_api_helper(parameter_a, value_b)
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]


@pytest.mark.parametrize(
    ("operator", "expected_type"),
    [
        (operator.add, Type.f32),
        (operator.sub, Type.f32),
        (operator.mul, Type.f32),
        (operator.truediv, Type.f32),
        (operator.eq, Type.boolean),
        (operator.ne, Type.boolean),
        (operator.gt, Type.boolean),
        (operator.ge, Type.boolean),
        (operator.lt, Type.boolean),
        (operator.le, Type.boolean),
    ],
)
def test_binary_operators(operator, expected_type):
    value_b = np.array([[4, 5], [1, 7]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == expected_type


@pytest.mark.parametrize(
    ("operator", "expected_type"),
    [
        (operator.add, Type.f32),
        (operator.sub, Type.f32),
        (operator.mul, Type.f32),
        (operator.truediv, Type.f32),
        (operator.eq, Type.boolean),
        (operator.ne, Type.boolean),
        (operator.gt, Type.boolean),
        (operator.ge, Type.boolean),
        (operator.lt, Type.boolean),
        (operator.le, Type.boolean),
    ],
)
def test_binary_operators_with_scalar(operator, expected_type):
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [2, 2]
    assert model.get_output_element_type(0) == expected_type


def test_multiply():
    A = np.arange(48, dtype=np.int32).reshape((8, 1, 6, 1))
    B = np.arange(35, dtype=np.int32).reshape((7, 1, 5))

    node = ng.multiply(A, B)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [8, 7, 6, 5]
    assert node.get_output_element_type(0) == Type.i32


def test_power_v1():
    A = np.arange(48, dtype=np.float32).reshape((8, 1, 6, 1))
    B = np.arange(20, dtype=np.float32).reshape((4, 1, 5))

    node = ng.power(A, B)

    assert node.get_type_name() == "Power"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [8, 4, 6, 5]
    assert node.get_output_element_type(0) == Type.f32
