# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset9 as ov
from openvino.runtime import Shape, Type
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node


@pytest.mark.parametrize(
    ("ng_api_fn", "numpy_fn", "range_start", "range_end"),
    [
        (ov.absolute, np.abs, -1, 1),
        (ov.abs, np.abs, -1, 1),
        (ov.acos, np.arccos, -1, 1),
        (ov.acosh, np.arccosh, 1, 2),
        (ov.asin, np.arcsin, -1, 1),
        (ov.asinh, np.arcsinh, -1, 1),
        (ov.atan, np.arctan, -100.0, 100.0),
        (ov.atanh, np.arctanh, 0.0, 1.0),
        (ov.ceiling, np.ceil, -100.0, 100.0),
        (ov.ceil, np.ceil, -100.0, 100.0),
        (ov.cos, np.cos, -100.0, 100.0),
        (ov.cosh, np.cosh, -100.0, 100.0),
        (ov.exp, np.exp, -100.0, 100.0),
        (ov.floor, np.floor, -100.0, 100.0),
        (ov.log, np.log, 0, 100.0),
        (ov.relu, lambda x: np.maximum(0, x), -100.0, 100.0),
        (ov.sign, np.sign, -100.0, 100.0),
        (ov.sin, np.sin, -100.0, 100.0),
        (ov.sinh, np.sinh, -100.0, 100.0),
        (ov.sqrt, np.sqrt, 0.0, 100.0),
        (ov.tan, np.tan, -1.0, 1.0),
        (ov.tanh, np.tanh, -100.0, 100.0),
    ],
)
def test_unary_op_array(ng_api_fn, numpy_fn, range_start, range_end):
    np.random.seed(133391)
    input_data = (range_start + np.random.rand(2, 3, 4) * (range_end - range_start)).astype(np.float32)
    expected = numpy_fn(input_data)

    result = run_op_node([input_data], ng_api_fn)
    assert np.allclose(result, expected, rtol=0.001)


@pytest.mark.parametrize(
    ("ng_api_fn", "numpy_fn", "input_data"),
    [
        pytest.param(ov.absolute, np.abs, np.float32(-3)),
        pytest.param(ov.abs, np.abs, np.float32(-3)),
        pytest.param(ov.acos, np.arccos, np.float32(-0.5)),
        pytest.param(ov.asin, np.arcsin, np.float32(-0.5)),
        pytest.param(ov.atan, np.arctan, np.float32(-0.5)),
        pytest.param(ov.ceiling, np.ceil, np.float32(1.5)),
        pytest.param(ov.ceil, np.ceil, np.float32(1.5)),
        pytest.param(ov.cos, np.cos, np.float32(np.pi / 4.0)),
        pytest.param(ov.cosh, np.cosh, np.float32(np.pi / 4.0)),
        pytest.param(ov.exp, np.exp, np.float32(1.5)),
        pytest.param(ov.floor, np.floor, np.float32(1.5)),
        pytest.param(ov.log, np.log, np.float32(1.5)),
        pytest.param(ov.relu, lambda x: np.maximum(0, x), np.float32(-0.125)),
        pytest.param(ov.sign, np.sign, np.float32(0.0)),
        pytest.param(ov.sin, np.sin, np.float32(np.pi / 4.0)),
        pytest.param(ov.sinh, np.sinh, np.float32(0.0)),
        pytest.param(ov.sqrt, np.sqrt, np.float32(3.5)),
        pytest.param(ov.tan, np.tan, np.float32(np.pi / 4.0)),
        pytest.param(ov.tanh, np.tanh, np.float32(0.1234)),
    ],
)
def test_unary_op_scalar(ng_api_fn, numpy_fn, input_data):
    expected = numpy_fn(input_data)

    result = run_op_node([input_data], ng_api_fn)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "input_data", [(np.array([True, False, True, False])), (np.array([True])), (np.array([False]))],
)
def test_logical_not(input_data):
    expected = np.logical_not(input_data)

    result = run_op_node([input_data], ov.logical_not)
    assert np.allclose(result, expected)


def test_sigmoid():
    input_data = np.array([-3.14, -1.0, 0.0, 2.71001, 1000.0], dtype=np.float32)
    result = run_op_node([input_data], ov.sigmoid)

    def sigmoid(value):
        return 1.0 / (1.0 + np.exp(-value))

    expected = np.array(list(map(sigmoid, input_data)))

    assert np.allclose(result, expected)


def test_softmax_positive_axis():
    axis = 1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    result = run_op_node([input_tensor], ov.softmax, axis)

    expected = [[0.09003056, 0.24472842, 0.6652409], [0.09003056, 0.24472842, 0.6652409]]

    assert np.allclose(result, expected)


def test_softmax_negative_axis():
    axis = -1
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    result = run_op_node([input_tensor], ov.softmax, axis)

    expected = [[0.09003056, 0.24472842, 0.6652409], [0.09003056, 0.24472842, 0.6652409]]

    assert np.allclose(result, expected)


def test_erf():
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.5, 3.14, 4.0], dtype=np.float32)
    expected = [-0.842701, 0.0, 0.842701, 0.999593, 0.999991, 1.0]

    result = run_op_node([input_tensor], ov.erf)
    assert np.allclose(result, expected)


def test_hswish():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.hswish(data)
    assert node.get_type_name() == "HSwish"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_round_even():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.round(data, "HALF_TO_EVEN")
    assert node.get_type_name() == "Round"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32

    input_tensor = np.array([-2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5], dtype=np.float32)
    expected = [-2.0, -2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 4.0]

    result = run_op_node([input_tensor], ov.round, "HALF_TO_EVEN")
    assert np.allclose(result, expected)


def test_round_away():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.round(data, "HALF_AWAY_FROM_ZERO")
    assert node.get_type_name() == "Round"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32

    input_tensor = np.array([-2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5], dtype=np.float32)
    expected = [-3.0, -2.0, -1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0]

    result = run_op_node([input_tensor], ov.round, "HALF_AWAY_FROM_ZERO")
    assert np.allclose(result, expected)


def test_hsigmoid():
    float_dtype = np.float32
    data = ov.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ov.hsigmoid(data)
    assert node.get_type_name() == "HSigmoid"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32


def test_gelu_operator_with_parameters():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    data_shape = [2, 2]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.gelu(parameter_data, "erf")
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array([[-1.6391277e-06, 8.4134471e-01], [-4.5500278e-02, 2.9959502]], dtype=np.float32)
    assert np.allclose(result, expected, 1e-6, 1e-6)


def test_gelu_operator_with_array():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ov.gelu(data_value, "erf")
    computation = runtime.computation(model)

    result = computation()
    expected = np.array([[-1.6391277e-06, 8.4134471e-01], [-4.5500278e-02, 2.9959502]], dtype=np.float32)
    assert np.allclose(result, expected, 1e-6, 1e-6)


def test_gelu_tanh_operator_with_parameters():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    data_shape = [2, 2]
    parameter_data = ov.parameter(data_shape, name="Data", dtype=np.float32)

    model = ov.gelu(parameter_data, "tanh")
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array([[0.0, 0.841192], [-0.04540223, 2.9963627]], dtype=np.float32)
    assert np.allclose(result, expected, 1e-6, 1e-6)


def test_gelu_tanh_operator_with_array():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ov.gelu(data_value, "tanh")
    computation = runtime.computation(model)

    result = computation()
    expected = np.array([[0.0, 0.841192], [-0.04540223, 2.9963627]], dtype=np.float32)

    assert np.allclose(result, expected, 1e-6, 1e-6)


@pytest.mark.parametrize(
    "data_type",
    [
        Type.f64,
        Type.f32,
        Type.f16,
    ],
)
def test_softsign_with_parameters(data_type):
    data = np.random.rand(4, 2).astype(data_type.to_dtype())
    expected = np.divide(data, np.abs(data) + 1)

    runtime = get_runtime()
    param = ov.parameter(data.shape, data_type, name="Data")
    result = runtime.computation(ov.softsign(param, "SoftSign"), param)(data)

    assert np.allclose(result, expected, 1e-6, 1e-3)


@pytest.mark.parametrize(
    "data_type",
    [
        np.float64,
        np.float32,
        np.float16,
    ],
)
def test_softsign_with_array(data_type):
    data = np.random.rand(32, 5).astype(data_type)
    expected = np.divide(data, np.abs(data) + 1)

    runtime = get_runtime()
    result = runtime.computation(ov.softsign(data, "SoftSign"))()

    assert np.allclose(result, expected, 1e-6, 1e-6)
