# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Shape, Type

R_TOLERANCE = 1e-6  # global relative tolerance


@pytest.mark.parametrize(
    ("graph_api_fn", "type_name"),
    [
        (ng.absolute, "Abs"),
        (ng.abs, "Abs"),
        (ng.acos, "Acos"),
        (ng.acosh, "Acosh"),
        (ng.asin, "Asin"),
        (ng.asinh, "Asinh"),
        (ng.atan, "Atan"),
        (ng.atanh, "Atanh"),
        (ng.ceiling, "Ceiling"),
        (ng.ceil, "Ceiling"),
        (ng.cos, "Cos"),
        (ng.cosh, "Cosh"),
        (ng.exp, "Exp"),
        (ng.floor, "Floor"),
        (ng.log, "Log"),
        (ng.relu, "Relu"),
        (ng.sign, "Sign"),
        (ng.sin, "Sin"),
        (ng.sinh, "Sinh"),
        (ng.sqrt, "Sqrt"),
        (ng.tan, "Tan"),
        (ng.tanh, "Tanh"),
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


@pytest.mark.parametrize(
    ("graph_api_fn", "input_data"),
    [
        pytest.param(ng.absolute, np.float32(-3)),
        pytest.param(ng.abs, np.float32(-3)),
        pytest.param(ng.acos, np.float32(-0.5)),
        pytest.param(ng.asin, np.float32(-0.5)),
        pytest.param(ng.atan, np.float32(-0.5)),
        pytest.param(ng.ceiling, np.float32(1.5)),
        pytest.param(ng.ceil, np.float32(1.5)),
        pytest.param(ng.cos, np.float32(np.pi / 4.0)),
        pytest.param(ng.cosh, np.float32(np.pi / 4.0)),
        pytest.param(ng.exp, np.float32(1.5)),
        pytest.param(ng.floor, np.float32(1.5)),
        pytest.param(ng.log, np.float32(1.5)),
        pytest.param(ng.relu, np.float32(-0.125)),
        pytest.param(ng.sign, np.float32(0.0)),
        pytest.param(ng.sin, np.float32(np.pi / 4.0)),
        pytest.param(ng.sinh, np.float32(0.0)),
        pytest.param(ng.sqrt, np.float32(3.5)),
        pytest.param(ng.tan, np.float32(np.pi / 4.0)),
        pytest.param(ng.tanh, np.float32(0.1234)),
    ],
)
def test_unary_op_scalar(graph_api_fn, input_data):
    node = graph_api_fn(input_data)

    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == []


@pytest.mark.parametrize(
    "input_data", [(np.array([True, False, True, False])), (np.array([True])), (np.array([False]))]
)
def test_logical_not(input_data):
    node = ng.logical_not(input_data)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "LogicalNot"
    assert node.get_output_element_type(0) == Type.boolean
    assert list(node.get_output_shape(0)) == list(input_data.shape)


def test_sigmoid():
    input_data = np.array([-3.14, -1.0, 0.0, 2.71001, 1000.0], dtype=np.float32)
    node = ng.sigmoid(input_data)

    assert node.get_output_size() == 1
    assert node.get_type_name() == "Sigmoid"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [5]


def test_softmax():
    axis = 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    node = ng.softmax(input_tensor, axis)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Softmax"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [2, 3]


def test_erf():
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.5, 3.14, 4.0], dtype=np.float32)
    node = ng.erf(input_tensor)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Erf"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [6]


def test_hswish():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ng.hswish(data)
    assert node.get_type_name() == "HSwish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_round_even():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ng.round(data, "HALF_TO_EVEN")
    assert node.get_type_name() == "Round"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_hsigmoid():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ng.hsigmoid(data)
    assert node.get_type_name() == "HSigmoid"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_gelu_operator_with_parameters():
    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.gelu(parameter_data, "erf")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_operator_with_array():
    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ng.gelu(data_value, "erf")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_tanh_operator_with_parameters():
    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name="Data", dtype=np.float32)

    model = ng.gelu(parameter_data, "tanh")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]


def test_gelu_tanh_operator_with_array():
    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ng.gelu(data_value, "tanh")
    assert model.get_output_size() == 1
    assert model.get_type_name() == "Gelu"
    assert model.get_output_element_type(0) == Type.f32
    assert list(model.get_output_shape(0)) == [2, 2]
