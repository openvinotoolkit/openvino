# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import operator

import numpy as np
import pytest

import ngraph as ng
from tests.test_ngraph.util import get_runtime, run_op_node


@pytest.mark.parametrize(
    "ng_api_helper,numpy_function",
    [
        (ng.add, np.add),
        (ng.divide, np.divide),
        (ng.multiply, np.multiply),
        (ng.subtract, np.subtract),
        (ng.minimum, np.minimum),
        (ng.maximum, np.maximum),
        (ng.mod, np.mod),
        (ng.equal, np.equal),
        (ng.not_equal, np.not_equal),
        (ng.greater, np.greater),
        (ng.greater_equal, np.greater_equal),
        (ng.less, np.less),
        (ng.less_equal, np.less_equal),
    ],
)
def test_binary_op(ng_api_helper, numpy_function):
    runtime = get_runtime()

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)
    parameter_b = ng.parameter(shape, name="B", dtype=np.float32)

    model = ng_api_helper(parameter_a, parameter_b)
    computation = runtime.computation(model, parameter_a, parameter_b)

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    result = computation(value_a, value_b)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "ng_api_helper,numpy_function",
    [
        (ng.add, np.add),
        (ng.divide, np.divide),
        (ng.multiply, np.multiply),
        (ng.subtract, np.subtract),
        (ng.minimum, np.minimum),
        (ng.maximum, np.maximum),
        (ng.mod, np.mod),
        (ng.equal, np.equal),
        (ng.not_equal, np.not_equal),
        (ng.greater, np.greater),
        (ng.greater_equal, np.greater_equal),
        (ng.less, np.less),
        (ng.less_equal, np.less_equal),
    ],
)
def test_binary_op_with_scalar(ng_api_helper, numpy_function):
    runtime = get_runtime()

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = ng_api_helper(parameter_a, value_b)
    computation = runtime.computation(model, parameter_a)

    result = computation(value_a)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "ng_api_helper,numpy_function",
    [(ng.logical_and, np.logical_and), (ng.logical_or, np.logical_or), (ng.logical_xor, np.logical_xor),],
)
def test_binary_logical_op(ng_api_helper, numpy_function):
    runtime = get_runtime()

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.bool)
    parameter_b = ng.parameter(shape, name="B", dtype=np.bool)

    model = ng_api_helper(parameter_a, parameter_b)
    computation = runtime.computation(model, parameter_a, parameter_b)

    value_a = np.array([[True, False], [False, True]], dtype=np.bool)
    value_b = np.array([[False, True], [False, True]], dtype=np.bool)

    result = computation(value_a, value_b)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "ng_api_helper,numpy_function",
    [(ng.logical_and, np.logical_and), (ng.logical_or, np.logical_or), (ng.logical_xor, np.logical_xor),],
)
def test_binary_logical_op_with_scalar(ng_api_helper, numpy_function):
    runtime = get_runtime()

    value_a = np.array([[True, False], [False, True]], dtype=np.bool)
    value_b = np.array([[False, True], [False, True]], dtype=np.bool)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.bool)

    model = ng_api_helper(parameter_a, value_b)
    computation = runtime.computation(model, parameter_a)

    result = computation(value_a)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "operator,numpy_function",
    [
        (operator.add, np.add),
        (operator.sub, np.subtract),
        (operator.mul, np.multiply),
        (operator.truediv, np.divide),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.lt, np.less),
        (operator.le, np.less_equal),
    ],
)
def test_binary_operators(operator, numpy_function):
    runtime = get_runtime()

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[4, 5], [1, 7]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)
    computation = runtime.computation(model, parameter_a)

    result = computation(value_a)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "operator,numpy_function",
    [
        (operator.add, np.add),
        (operator.sub, np.subtract),
        (operator.mul, np.multiply),
        (operator.truediv, np.divide),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.lt, np.less),
        (operator.le, np.less_equal),
    ],
)
def test_binary_operators_with_scalar(operator, numpy_function):
    runtime = get_runtime()

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name="A", dtype=np.float32)

    model = operator(parameter_a, value_b)
    computation = runtime.computation(model, parameter_a)

    result = computation(value_a)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)


def test_multiply():
    A = np.arange(48).reshape((8, 1, 6, 1))
    B = np.arange(35).reshape((7, 1, 5))

    expected = np.multiply(A, B)
    result = run_op_node([A, B], ng.multiply)

    assert np.allclose(result, expected)


def test_power_v1():
    A = np.arange(48, dtype=np.float32).reshape((8, 1, 6, 1))
    B = np.arange(20, dtype=np.float32).reshape((4, 1, 5))

    expected = np.power(A, B)
    result = run_op_node([A, B], ng.power)

    assert np.allclose(result, expected)
