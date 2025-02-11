# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset13 as ops
from openvino import Shape, Type

R_TOLERANCE = 1e-6  # global relative tolerance


@pytest.mark.parametrize(
    ("graph_api_fn", "type_name"),
    [
        (ops.absolute, "Abs"),
        (ops.abs, "Abs"),
        (ops.acos, "Acos"),
        (ops.acosh, "Acosh"),
        (ops.asin, "Asin"),
        (ops.asinh, "Asinh"),
        (ops.atan, "Atan"),
        (ops.atanh, "Atanh"),
        (ops.ceiling, "Ceiling"),
        (ops.ceil, "Ceiling"),
        (ops.cos, "Cos"),
        (ops.cosh, "Cosh"),
        (ops.exp, "Exp"),
        (ops.floor, "Floor"),
        (ops.log, "Log"),
        (ops.relu, "Relu"),
        (ops.sign, "Sign"),
        (ops.sin, "Sin"),
        (ops.sinh, "Sinh"),
        (ops.sqrt, "Sqrt"),
        (ops.tan, "Tan"),
        (ops.tanh, "Tanh"),
    ],
)
def test_unary_op_array(graph_api_fn, type_name):
    np.random.seed(133391)
    input_data = np.random.rand(2, 3, 4).astype(np.float32)

    node = graph_api_fn(input_data)
    assert node.get_output_size() == 1
    assert node.get_type_name() == type_name
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [2, 3, 4]


@pytest.mark.parametrize("graph_api_fn", [
    ops.absolute,
    ops.abs,
    ops.acos,
    ops.asin,
    ops.atan,
    ops.ceiling,
    ops.ceil,
    ops.cos,
    ops.cosh,
    ops.exp,
    ops.floor,
    ops.log,
    ops.relu,
    ops.sign,
    ops.sin,
    ops.sinh,
    ops.sqrt,
    ops.tan,
    ops.tanh,
])
def test_unary_op_scalar(graph_api_fn):
    node = graph_api_fn(np.float32(-0.5))

    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == []


@pytest.mark.parametrize(
    "input_data",
    [(np.array([True, False, True, False])), (np.array([True])), (np.array([False]))],
)
def test_logical_not(input_data):
    node = ops.logical_not(input_data)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "LogicalNot"
    assert node.get_output_element_type(0) == Type.boolean
    assert list(node.get_output_shape(0)) == list(input_data.shape)


def test_sigmoid():
    input_data = np.array([-3.14, -1.0, 0.0, 2.71001, 1000.0], dtype=np.float32)
    node = ops.sigmoid(input_data)

    assert node.get_output_size() == 1
    assert node.get_type_name() == "Sigmoid"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [5]


def test_softmax():
    axis = 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    node = ops.softmax(input_tensor, axis)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Softmax"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [2, 3]


def test_erf():
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.5, 3.14, 4.0], dtype=np.float32)

    node = ops.erf(input_tensor)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Erf"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [6]


def test_hswish():
    float_dtype = np.float32
    data = ops.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ops.hswish(data)
    assert node.get_type_name() == "HSwish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_round():
    float_dtype = np.float32
    data = ops.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ops.round(data, "HALF_TO_EVEN")
    assert node.get_type_name() == "Round"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32

    input_tensor = np.array([-2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5], dtype=np.float32)

    node = ops.round(input_tensor, "HALF_TO_EVEN")
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Round"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [9]


def test_hsigmoid():
    float_dtype = np.float32
    data = ops.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ops.hsigmoid(data)
    assert node.get_type_name() == "HSigmoid"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_gelu_operator_with_parameters():
    data_shape = [2, 2]
    parameter_data = ops.parameter(data_shape, name="Data", dtype=np.float32)

    model = ops.gelu(parameter_data, "erf")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_operator_with_array():
    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ops.gelu(data_value, "erf")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_tanh_operator_with_parameters():
    data_shape = [2, 2]
    parameter_data = ops.parameter(data_shape, name="Data", dtype=np.float32)

    model = ops.gelu(parameter_data, "tanh")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_tanh_operator_with_array():
    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ops.gelu(data_value, "tanh")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


@pytest.mark.parametrize(
    ("input_data", "dtype"),
    [
        (np.array([True, False, True, False]), Type.boolean),
        (np.array([True]), Type.boolean),
        (np.array([False]), Type.boolean),
        (np.array([0, 3, 7, 256], dtype=np.uint16), Type.u16),
        (np.array([[-7, 0], [256, 1]], dtype=np.int32), Type.i32),
    ],
)
def test_bitwise_not(input_data, dtype):
    node = ops.bitwise_not(input_data)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "BitwiseNot"
    assert node.get_output_element_type(0) == dtype
    assert list(node.get_output_shape(0)) == list(input_data.shape)
