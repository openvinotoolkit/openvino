# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime as ov_runtime
import openvino.runtime.opset13 as ov
from openvino.runtime import Shape, Type

R_TOLERANCE = 1e-6  # global relative tolerance


@pytest.mark.parametrize(
    ("graph_api_fn", "type_name"),
    [
        (ov.absolute, "Abs"),
        (ov.abs, "Abs"),
        (ov.acos, "Acos"),
        (ov.acosh, "Acosh"),
        (ov.asin, "Asin"),
        (ov.asinh, "Asinh"),
        (ov.atan, "Atan"),
        (ov.atanh, "Atanh"),
        (ov.ceiling, "Ceiling"),
        (ov.ceil, "Ceiling"),
        (ov.cos, "Cos"),
        (ov.cosh, "Cosh"),
        (ov.exp, "Exp"),
        (ov.floor, "Floor"),
        (ov.log, "Log"),
        (ov.relu, "Relu"),
        (ov.sign, "Sign"),
        (ov.sin, "Sin"),
        (ov.sinh, "Sinh"),
        (ov.sqrt, "Sqrt"),
        (ov.tan, "Tan"),
        (ov.tanh, "Tanh"),
    ],
)
def test_unary_op_array(graph_api_fn, type_name):
    np.random.seed(133391)
    input_data = np.random.rand(2, 3, 4).astype(np.float32)

    node = graph_api_fn(input_data)
    assert node.get_output_size() == 1
    assert node.get_type_name() == type_name
    assert node.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(node.get_output_shape(0)) == [2, 3, 4]


@pytest.mark.parametrize("graph_api_fn", [
    ov.absolute,
    ov.abs,
    ov.acos,
    ov.asin,
    ov.atan,
    ov.ceiling,
    ov.ceil,
    ov.cos,
    ov.cosh,
    ov.exp,
    ov.floor,
    ov.log,
    ov.relu,
    ov.sign,
    ov.sin,
    ov.sinh,
    ov.sqrt,
    ov.tan,
    ov.tanh,
])
def test_unary_op_scalar(graph_api_fn):
    node = graph_api_fn(np.float32(-0.5))

    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(node.get_output_shape(0)) == []


@pytest.mark.parametrize(
    "input_data",
    [(np.array([True, False, True, False])), (np.array([True])), (np.array([False]))],
)
def test_logical_not(input_data):
    node = ov.logical_not(input_data)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "LogicalNot"
    assert node.get_output_element_type(0) == ov_runtime.Type.boolean
    assert list(node.get_output_shape(0)) == list(input_data.shape)


def test_sigmoid():
    input_data = np.array([-3.14, -1.0, 0.0, 2.71001, 1000.0], dtype=np.float32)
    node = ov.sigmoid(input_data)

    assert node.get_output_size() == 1
    assert node.get_type_name() == "Sigmoid"
    assert node.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(node.get_output_shape(0)) == [5]


def test_softmax():
    axis = 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    node = ov.softmax(input_tensor, axis)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Softmax"
    assert node.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(node.get_output_shape(0)) == [2, 3]


def test_erf():
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.5, 3.14, 4.0], dtype=np.float32)

    node = ov.erf(input_tensor)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Erf"
    assert node.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(node.get_output_shape(0)) == [6]


def test_hswish():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.hswish(data)
    assert node.get_type_name() == "HSwish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_round():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.round(data, "HALF_TO_EVEN")
    assert node.get_type_name() == "Round"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32

    input_tensor = np.array([-2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5], dtype=np.float32)

    node = ov.round(input_tensor, "HALF_TO_EVEN")
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Round"
    assert node.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(node.get_output_shape(0)) == [9]


def test_hsigmoid():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.hsigmoid(data)
    assert node.get_type_name() == "HSigmoid"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_gelu_operator_with_parameters():
    data_shape = [2, 2]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.gelu(parameter_data, "erf")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_operator_with_array():
    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ov.gelu(data_value, "erf")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_tanh_operator_with_parameters():
    data_shape = [2, 2]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.gelu(parameter_data, "tanh")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_tanh_operator_with_array():
    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ov.gelu(data_value, "tanh")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == ov_runtime.Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


@pytest.mark.parametrize(
    ("input_data", "dtype"),
    [
        (np.array([True, False, True, False]), ov_runtime.Type.boolean),
        (np.array([True]), ov_runtime.Type.boolean),
        (np.array([False]), ov_runtime.Type.boolean),
        (np.array([0, 3, 7, 256], dtype=np.uint16), ov_runtime.Type.u16),
        (np.array([[-7, 0], [256, 1]], dtype=np.int32), ov_runtime.Type.i32),
    ],
)
def test_bitwise_not(input_data, dtype):
    node = ov.bitwise_not(input_data)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "BitwiseNot"
    assert node.get_output_element_type(0) == dtype
    assert list(node.get_output_shape(0)) == list(input_data.shape)
