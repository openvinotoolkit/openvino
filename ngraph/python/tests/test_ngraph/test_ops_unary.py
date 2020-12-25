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
import numpy as np
import pytest

import ngraph as ng
from ngraph.impl import Shape, Type
from tests.test_ngraph.util import run_op_node
from tests import xfail_issue_44970


@pytest.mark.parametrize(
    "ng_api_fn, numpy_fn, range_start, range_end",
    [
        (ng.absolute, np.abs, -1, 1),
        (ng.abs, np.abs, -1, 1),
        (ng.acos, np.arccos, -1, 1),
        (ng.acosh, np.arccosh, 1, 2),
        (ng.asin, np.arcsin, -1, 1),
        (ng.asinh, np.arcsinh, -1, 1),
        (ng.atan, np.arctan, -100.0, 100.0),
        (ng.atanh, np.arctanh, 0.0, 1.0),
        (ng.ceiling, np.ceil, -100.0, 100.0),
        (ng.ceil, np.ceil, -100.0, 100.0),
        (ng.cos, np.cos, -100.0, 100.0),
        (ng.cosh, np.cosh, -100.0, 100.0),
        (ng.exp, np.exp, -100.0, 100.0),
        (ng.floor, np.floor, -100.0, 100.0),
        (ng.log, np.log, 0, 100.0),
        (ng.relu, lambda x: np.maximum(0, x), -100.0, 100.0),
        (ng.sign, np.sign, -100.0, 100.0),
        (ng.sin, np.sin, -100.0, 100.0),
        (ng.sinh, np.sinh, -100.0, 100.0),
        (ng.sqrt, np.sqrt, 0.0, 100.0),
        (ng.tan, np.tan, -1.0, 1.0),
        (ng.tanh, np.tanh, -100.0, 100.0),
    ],
)
def test_unary_op_array(ng_api_fn, numpy_fn, range_start, range_end):
    np.random.seed(133391)
    input_data = (range_start + np.random.rand(2, 3, 4) * (range_end - range_start)).astype(np.float32)
    expected = numpy_fn(input_data)

    result = run_op_node([input_data], ng_api_fn)
    assert np.allclose(result, expected, rtol=0.001)


@pytest.mark.parametrize(
    "ng_api_fn, numpy_fn, input_data",
    [
        pytest.param(ng.absolute, np.abs, np.float32(-3)),
        pytest.param(ng.abs, np.abs, np.float32(-3)),
        pytest.param(ng.acos, np.arccos, np.float32(-0.5)),
        pytest.param(ng.asin, np.arcsin, np.float32(-0.5)),
        pytest.param(ng.atan, np.arctan, np.float32(-0.5)),
        pytest.param(ng.ceiling, np.ceil, np.float32(1.5)),
        pytest.param(ng.ceil, np.ceil, np.float32(1.5)),
        pytest.param(ng.cos, np.cos, np.float32(np.pi / 4.0)),
        pytest.param(ng.cosh, np.cosh, np.float32(np.pi / 4.0)),
        pytest.param(ng.exp, np.exp, np.float32(1.5)),
        pytest.param(ng.floor, np.floor, np.float32(1.5)),
        pytest.param(ng.log, np.log, np.float32(1.5)),
        pytest.param(ng.relu, lambda x: np.maximum(0, x), np.float32(-0.125)),
        pytest.param(ng.sign, np.sign, np.float32(0.0)),
        pytest.param(ng.sin, np.sin, np.float32(np.pi / 4.0)),
        pytest.param(ng.sinh, np.sinh, np.float32(0.0)),
        pytest.param(ng.sqrt, np.sqrt, np.float32(3.5)),
        pytest.param(ng.tan, np.tan, np.float32(np.pi / 4.0)),
        pytest.param(ng.tanh, np.tanh, np.float32(0.1234)),
    ],
)
def test_unary_op_scalar(ng_api_fn, numpy_fn, input_data):
    expected = numpy_fn(input_data)

    result = run_op_node([input_data], ng_api_fn)
    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "input_data", [(np.array([True, False, True, False])), (np.array([True])), (np.array([False]))]
)
def test_logical_not(input_data):
    expected = np.logical_not(input_data)

    result = run_op_node([input_data], ng.logical_not)
    assert np.allclose(result, expected)


def test_sigmoid():
    input_data = np.array([-3.14, -1.0, 0.0, 2.71001, 1000.0], dtype=np.float32)
    result = run_op_node([input_data], ng.sigmoid)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    expected = np.array(list(map(sigmoid, input_data)))

    assert np.allclose(result, expected)


@xfail_issue_44970
def test_softmax():
    axis = 0
    input_tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    result = run_op_node([input_tensor], ng.softmax, axis)

    expected = [[0.00426978, 0.01160646, 0.03154963], [0.08576079, 0.23312202, 0.6336913]]

    assert np.allclose(result, expected)


def test_erf():
    input_tensor = np.array([-1.0, 0.0, 1.0, 2.5, 3.14, 4.0], dtype=np.float32)
    expected = [-0.842701, 0.0, 0.842701, 0.999593, 0.999991, 1.0]

    result = run_op_node([input_tensor], ng.erf)
    assert np.allclose(result, expected)


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

    input_tensor = np.array([-2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5], dtype=np.float32)
    expected = [-2.0, -2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 4.0]

    result = run_op_node([input_tensor], ng.round, "HALF_TO_EVEN")
    assert np.allclose(result, expected)


def test_round_away():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ng.round(data, "HALF_AWAY_FROM_ZERO")
    assert node.get_type_name() == "Round"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32

    input_tensor = np.array([-2.5, -1.5, -0.5, 0.5, 0.9, 1.5, 2.3, 2.5, 3.5], dtype=np.float32)
    expected = [-3.0, -2.0, -1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0]

    result = run_op_node([input_tensor], ng.round, "HALF_AWAY_FROM_ZERO")
    assert np.allclose(result, expected)


def test_hsigmoid():
    float_dtype = np.float32
    data = ng.parameter(Shape([3, 10]), dtype=float_dtype, name="data")

    node = ng.hsigmoid(data)
    assert node.get_type_name() == "HSigmoid"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 10]
    assert node.get_output_element_type(0) == Type.f32
