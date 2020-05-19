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
# flake8: noqa
from __future__ import absolute_import

import pytest
import numpy as np
import ngraph as ng

from ngraph.impl import util
from ngraph.impl import Shape, Strides, CoordinateDiff, AxisSet, Coordinate
from ngraph.impl import Type, Function
from ngraph.impl.runtime import Backend
from ngraph.impl.op import Parameter
from ngraph.impl.op import Constant

import test


def binary_op(op_str, a, b):

    if op_str == "+":
        return a + b
    elif op_str == "Add":
        return ng.add(a, b)
    elif op_str == "-":
        return a - b
    elif op_str == "Sub":
        return ng.subtract(a, b)
    elif op_str == "*":
        return a * b
    elif op_str == "Mul":
        return ng.multiply(a, b)
    elif op_str == "/":
        return a / b
    elif op_str == "Div":
        return ng.divide(a, b)
    elif op_str == "Equal":
        return ng.equal(a, b)
    elif op_str == "Greater":
        return ng.greater(a, b)
    elif op_str == "GreaterEq":
        return ng.greater_equal(a, b)
    elif op_str == "Less":
        return ng.less(a, b)
    elif op_str == "LessEq":
        return ng.less_equal(a, b)
    elif op_str == "Maximum":
        return ng.maximum(a, b)
    elif op_str == "Minimum":
        return ng.minimum(a, b)
    elif op_str == "NotEqual":
        return ng.not_equal(a, b)
    elif op_str == "Power":
        return ng.power(a, b)


def binary_op_ref(op_str, a, b):

    if op_str == "+" or op_str == "Add":
        return a + b
    elif op_str == "-" or op_str == "Sub":
        return a - b
    elif op_str == "*" or op_str == "Mul":
        return a * b
    elif op_str == "/" or op_str == "Div":
        return a / b
    elif op_str == "Dot":
        return np.dot(a, b)
    elif op_str == "Equal":
        return np.equal(a, b)
    elif op_str == "Greater":
        return np.greater(a, b)
    elif op_str == "GreaterEq":
        return np.greater_equal(a, b)
    elif op_str == "Less":
        return np.less(a, b)
    elif op_str == "LessEq":
        return np.less_equal(a, b)
    elif op_str == "Maximum":
        return np.maximum(a, b)
    elif op_str == "Minimum":
        return np.minimum(a, b)
    elif op_str == "NotEqual":
        return np.not_equal(a, b)
    elif op_str == "Power":
        return np.power(a, b)


def binary_op_exec(op_str):

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    parameter_list = [A, B]
    function = Function([binary_op(op_str, A, B)], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, shape)
    b = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, shape)

    a.write(util.numpy_to_c(np.array([[1, 6], [7, 4]], dtype=np.float32)), 16)
    b.write(util.numpy_to_c(np.array([[5, 2], [3, 8]], dtype=np.float32)), 16)

    result_arr = np.array([[0, 0], [0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 16)
    handle = backend.compile(function)
    handle.call([result], [a, b])
    result.read(util.numpy_to_c(result_arr), 16)

    a_arr = np.array([[1, 6], [7, 4]], dtype=np.float32)
    b_arr = np.array([[5, 2], [3, 8]], dtype=np.float32)
    result_arr_ref = binary_op_ref(op_str, a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def binary_op_comparison(op_str):

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    parameter_list = [A, B]
    function = Function([binary_op(op_str, A, B)], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, shape)
    b = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(Type.boolean, shape)

    a.write(util.numpy_to_c(np.array([[1, 5], [3, 2]], dtype=np.float32)), 16)
    b.write(util.numpy_to_c(np.array([[2, 4], [3, 1]], dtype=np.float32)), 16)

    result_arr = np.array([[False, False], [False, False]], dtype=np.bool)
    result.write(util.numpy_to_c(result_arr), 4)
    handle = backend.compile(function)
    handle.call([result], [a, b])
    result.read(util.numpy_to_c(result_arr), 4)

    a_arr = np.array([[1, 5], [3, 2]], dtype=np.float32)
    b_arr = np.array([[2, 4], [3, 1]], dtype=np.float32)
    result_arr_ref = binary_op_ref(op_str, a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_add():
    binary_op_exec("+")


def test_add_op():
    binary_op_exec("Add")


def test_sub():
    binary_op_exec("-")


def test_sub_op():
    binary_op_exec("Sub")


def test_mul():
    binary_op_exec("*")


def test_mul_op():
    binary_op_exec("Mul")


def test_div():
    binary_op_exec("/")


def test_div_op():
    binary_op_exec("Div")


def test_maximum():
    binary_op_exec("Maximum")


def test_minimum():
    binary_op_exec("Minimum")


def test_power():
    binary_op_exec("Power")


def test_greater():
    binary_op_comparison("Greater")


def test_greater_eq():
    binary_op_comparison("GreaterEq")


def test_less():
    binary_op_comparison("Less")


def test_less_eq():
    binary_op_comparison("LessEq")


def test_not_equal():
    binary_op_comparison("NotEqual")


def test_add_with_mul():

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    C = Parameter(element_type, shape)
    parameter_list = [A, B, C]
    function = Function([ng.multiply(ng.add(A, B), C)], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, shape)
    b = backend.create_tensor(element_type, shape)
    c = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, shape)

    a.write(util.numpy_to_c(np.array([1, 2, 3, 4], dtype=np.float32)), 16)
    b.write(util.numpy_to_c(np.array([5, 6, 7, 8], dtype=np.float32)), 16)
    c.write(util.numpy_to_c(np.array([9, 10, 11, 12], dtype=np.float32)), 16)

    result_arr = np.array([0, 0, 0, 0], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 16)
    handle = backend.compile(function)
    handle.call([result], [a, b, c])
    result.read(util.numpy_to_c(result_arr), 16)

    a_arr = np.array([1, 2, 3, 4], dtype=np.float32)
    b_arr = np.array([5, 6, 7, 8], dtype=np.float32)
    c_arr = np.array([9, 10, 11, 12], dtype=np.float32)
    result_arr_ref = (a_arr + b_arr) * c_arr

    assert np.allclose(result_arr, result_arr_ref)


def unary_op(op_str, a):
    if op_str == "Abs":
        return ng.abs(a)
    elif op_str == "Acos":
        return ng.acos(a)
    elif op_str == "Asin":
        return ng.asin(a)
    elif op_str == "Atan":
        return ng.atan(a)
    elif op_str == "Ceiling":
        return ng.ceiling(a)
    elif op_str == "Cos":
        return ng.cos(a)
    elif op_str == "Cosh":
        return ng.cosh(a)
    elif op_str == "Floor":
        return ng.floor(a)
    elif op_str == "log":
        return ng.log(a)
    elif op_str == "exp":
        return ng.exp(a)
    elif op_str == "negative":
        return ng.negative(a)
    elif op_str == "Reverse":
        return ng.reverse(a, np.array([1]), "index")
    elif op_str == "Sign":
        return ng.sign(a)
    elif op_str == "Sin":
        return ng.sin(a)
    elif op_str == "Sinh":
        return ng.sinh(a)
    elif op_str == "Sqrt":
        return ng.sqrt(a)
    elif op_str == "Tan":
        return ng.tan(a)
    elif op_str == "Tanh":
        return ng.tanh(a)


def unary_op_ref(op_str, a):
    if op_str == "Abs":
        return np.abs(a)
    elif op_str == "Acos":
        return np.arccos(a)
    elif op_str == "Asin":
        return np.arcsin(a)
    elif op_str == "Atan":
        return np.arctan(a)
    elif op_str == "Ceiling":
        return np.ceil(a)
    elif op_str == "Cos":
        return np.cos(a)
    elif op_str == "Cosh":
        return np.cosh(a)
    elif op_str == "Floor":
        return np.floor(a)
    elif op_str == "log":
        return np.log(a)
    elif op_str == "exp":
        return np.exp(a)
    elif op_str == "negative":
        return np.negative(a)
    elif op_str == "Reverse":
        return np.fliplr(a)
    elif op_str == "Sign":
        return np.sign(a)
    elif op_str == "Sin":
        return np.sin(a)
    elif op_str == "Sinh":
        return np.sinh(a)
    elif op_str == "Sqrt":
        return np.sqrt(a)
    elif op_str == "Tan":
        return np.tan(a)
    elif op_str == "Tanh":
        return np.tanh(a)


def unary_op_exec(op_str, input_list):
    """
    input_list needs to have deep length of 4
    """
    element_type = Type.f32
    shape = Shape(np.array(input_list).shape)
    shape_np = np.array(input_list).shape
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function([unary_op(op_str, A)], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, shape)

    a.write(util.numpy_to_c(np.array(input_list, dtype=np.float32)), 16)

    result_arr = np.zeros(shape_np, dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 16)
    handle = backend.compile(function)
    handle.call([result], [a])
    result.read(util.numpy_to_c(result_arr), 16)

    a_arr = np.array(input_list, dtype=np.float32)
    result_arr_ref = unary_op_ref(op_str, a_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_abs():
    input_list = [-1, 0, 1, 2]
    op_str = "Abs"
    unary_op_exec(op_str, input_list)


def test_acos():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Acos"
    unary_op_exec(op_str, input_list)


def test_asin():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Asin"
    unary_op_exec(op_str, input_list)


def test_atan():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Atan"
    unary_op_exec(op_str, input_list)


def test_ceiling():
    input_list = [0.5, 0, 0.4, 0.5]
    op_str = "Ceiling"
    unary_op_exec(op_str, input_list)


def test_cos():
    input_list = [0, 0.7, 1.7, 3.4]
    op_str = "Cos"
    unary_op_exec(op_str, input_list)


def test_cosh():
    input_list = [-1, 0.0, 0.5, 1]
    op_str = "Cosh"
    unary_op_exec(op_str, input_list)


def test_floor():
    input_list = [-0.5, 0, 0.4, 0.5]
    op_str = "Floor"
    unary_op_exec(op_str, input_list)


def test_log():
    input_list = [1, 2, 3, 4]
    op_str = "log"
    unary_op_exec(op_str, input_list)


def test_exp():
    input_list = [-1, 0, 1, 2]
    op_str = "exp"
    unary_op_exec(op_str, input_list)


def test_negative():
    input_list = [-1, 0, 1, 2]
    op_str = "negative"
    unary_op_exec(op_str, input_list)


def test_sign():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Sign"
    unary_op_exec(op_str, input_list)


def test_sin():
    input_list = [0, 0.7, 1.7, 3.4]
    op_str = "Sin"
    unary_op_exec(op_str, input_list)


def test_sinh():
    input_list = [-1, 0.0, 0.5, 1]
    op_str = "Sinh"
    unary_op_exec(op_str, input_list)


def test_sqrt():
    input_list = [0.0, 0.5, 1, 2]
    op_str = "Sqrt"
    unary_op_exec(op_str, input_list)


def test_tan():
    input_list = [-np.pi / 4, 0, np.pi / 8, np.pi / 8]
    op_str = "Tan"
    unary_op_exec(op_str, input_list)


def test_tanh():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Tanh"
    unary_op_exec(op_str, input_list)


def test_reverse():
    input_list = [[-1, 0], [0.5, 1]]
    op_str = "Reverse"
    unary_op_exec(op_str, input_list)


def test_reshape():

    element_type = Type.f32
    shape = Shape([2, 3])
    A = Parameter(element_type, shape)
    parameter_list = [A]
    function = Function([ng.reshape(A, Shape([3, 2]), special_zero=False)], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([3, 2]))

    a.write(util.numpy_to_c(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)), 24)

    result_arr = np.array([[0, 0], [0, 0], [0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 24)
    handle = backend.compile(function)
    handle.call([result], [a])
    result.read(util.numpy_to_c(result_arr), 24)

    a_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    result_arr_ref = np.reshape(a_arr, (3, 2))

    assert np.allclose(result_arr, result_arr_ref)


def test_broadcast():

    element_type = Type.f32
    A = Parameter(element_type, Shape([3]))
    parameter_list = [A]
    function = Function([ng.broadcast(A, [3, 3])], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, Shape([3]))
    result = backend.create_tensor(element_type, Shape([3, 3]))

    a.write(util.numpy_to_c(np.array([1, 2, 3], dtype=np.float32)), 12)

    result_arr = np.zeros((3, 3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 36)
    handle = backend.compile(function)
    handle.call([result], [a])
    result.read(util.numpy_to_c(result_arr), 36)

    a_arr = np.array([[0], [0], [0]], dtype=np.float32)
    b_arr = np.array([[1, 2, 3]], dtype=np.float32)
    result_arr_ref = np.add(a_arr, b_arr)

    assert np.allclose(result_arr, result_arr_ref)


def test_constant():

    element_type = Type.f32
    parameter_list = []
    function = Function(
        [Constant(element_type, Shape([3, 3]), list(range(9)))], parameter_list, "test"
    )
    backend = Backend.create(test.BACKEND_NAME)

    result = backend.create_tensor(element_type, Shape([3, 3]))

    result_arr = np.zeros((3, 3), dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 36)
    handle = backend.compile(function)
    handle.call([result], [])
    result.read(util.numpy_to_c(result_arr), 36)

    result_arr_ref = np.arange(9).reshape(3, 3)

    assert np.allclose(result_arr, result_arr_ref)


def test_concat():

    element_type = Type.f32
    A = Parameter(element_type, Shape([1, 2]))
    B = Parameter(element_type, Shape([1, 2]))
    C = Parameter(element_type, Shape([1, 2]))
    parameter_list = [A, B, C]
    axis = 0
    function = Function([ng.concat([A, B, C], axis)], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, Shape([1, 2]))
    b = backend.create_tensor(element_type, Shape([1, 2]))
    c = backend.create_tensor(element_type, Shape([1, 2]))
    result = backend.create_tensor(element_type, Shape([3, 2]))

    a.write(util.numpy_to_c(np.array([1, 2], dtype=np.float32)), 8)
    b.write(util.numpy_to_c(np.array([5, 6], dtype=np.float32)), 8)
    c.write(util.numpy_to_c(np.array([7, 8], dtype=np.float32)), 8)

    result_arr = np.zeros(6, dtype=np.float32).reshape(3, 2)
    result.write(util.numpy_to_c(result_arr), 24)
    handle = backend.compile(function)
    handle.call([result], [a, b, c])
    result.read(util.numpy_to_c(result_arr), 24)

    a_arr = np.array([[1, 2]], dtype=np.float32)
    b_arr = np.array([[5, 6]], dtype=np.float32)
    c_arr = np.array([[7, 8]], dtype=np.float32)
    result_arr_ref = np.concatenate((a_arr, b_arr, c_arr), axis)

    assert np.allclose(result_arr, result_arr_ref)


def test_axisset():

    set_axisset = AxisSet({1, 2, 3})
    list_axisset = AxisSet([1, 2, 3])
    tuple_axisset = AxisSet((1, 2, 3))

    assert len(set_axisset) == 3
    assert set(set_axisset) == {1, 2, 3}

    assert len(list_axisset) == 3
    assert set(list_axisset) == set(set_axisset)

    assert len(tuple_axisset) == 3
    assert set(tuple_axisset) == set(set_axisset)


def test_select():

    element_type = Type.f32
    A = Parameter(Type.boolean, Shape([1, 2]))
    B = Parameter(element_type, Shape([1, 2]))
    C = Parameter(element_type, Shape([1, 2]))
    parameter_list = [A, B, C]

    function = Function([ng.select(A, B, C)], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(Type.boolean, Shape([1, 2]))
    b = backend.create_tensor(element_type, Shape([1, 2]))
    c = backend.create_tensor(element_type, Shape([1, 2]))
    result = backend.create_tensor(element_type, Shape([1, 2]))

    a.write(util.numpy_to_c(np.array([[True, False]], dtype=np.bool)), 2)
    b.write(util.numpy_to_c(np.array([[5, 6]], dtype=np.float32)), 8)
    c.write(util.numpy_to_c(np.array([[7, 8]], dtype=np.float32)), 8)

    result_arr = np.array([[0, 0]], dtype=np.float32)
    result.write(util.numpy_to_c(result_arr), 8)
    handle = backend.compile(function)
    handle.call([result], [a, b, c])
    result.read(util.numpy_to_c(result_arr), 8)

    result_arr_ref = np.array([[5, 8]])

    assert np.allclose(result_arr, result_arr_ref)


def test_max_pool():
    # test 1d
    element_type = Type.f32
    shape = Shape([1, 1, 10])
    A = Parameter(element_type, shape)
    parameter_list = [A]

    input_arr = np.arange(10, dtype=np.float32).reshape(1, 1, 10)
    window_shape = [3]

    strides = [1] * len(window_shape)
    pads_begin = [0] * len(window_shape)
    pads_end = [0] * len(window_shape)

    model = ng.max_pool(A, strides, pads_begin, pads_end, window_shape)
    function = Function([model], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([1, 1, 8]))

    a.write(util.numpy_to_c(input_arr), 10 * 4)

    result_arr = np.zeros(8, dtype=np.float32).reshape(1, 1, 8)
    result.write(util.numpy_to_c(result_arr), 8 * 4)
    handle = backend.compile(function)
    handle.call([result], [a])
    result.read(util.numpy_to_c(result_arr), 32)

    result_arr_ref = (np.arange(8) + 2).reshape(1, 1, 8)
    assert np.allclose(result_arr, result_arr_ref)

    # test 1d with strides
    strides = [2]
    pads_begin = [0] * len(window_shape)
    pads_end = [0] * len(window_shape)

    model = ng.max_pool(A, strides, pads_begin, pads_end, window_shape)
    function = Function([model], parameter_list, "test")

    size = 4
    result = backend.create_tensor(element_type, Shape([1, 1, size]))
    result_arr = np.zeros(size, dtype=np.float32).reshape(1, 1, size)

    backend = Backend.create(test.BACKEND_NAME)
    result.write(util.numpy_to_c(result_arr), size * 4)
    handle = backend.compile(function)
    handle.call([result], [a])
    result.read(util.numpy_to_c(result_arr), size * 4)

    result_arr_ref = ((np.arange(size) + 1) * 2).reshape(1, 1, size)
    assert np.allclose(result_arr, result_arr_ref)

    # test 2d
    element_type = Type.f32
    shape = Shape([1, 1, 10, 10])
    A = Parameter(element_type, shape)
    parameter_list = [A]

    input_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    window_shape = [3, 3]

    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ng.max_pool(A, strides, pads_begin, pads_end, window_shape)
    function = Function([model], parameter_list, "test")

    backend = Backend.create(test.BACKEND_NAME)
    a = backend.create_tensor(element_type, shape)
    result = backend.create_tensor(element_type, Shape([1, 1, 8, 8]))

    a.write(util.numpy_to_c(input_arr), 10 * 10 * 4)

    result_arr = np.zeros(64, dtype=np.float32).reshape(1, 1, 8, 8)
    result.write(util.numpy_to_c(result_arr), 8 * 8 * 4)
    handle = backend.compile(function)
    handle.call([result], [a])
    result.read(util.numpy_to_c(result_arr), 8 * 8 * 4)

    result_arr_ref = ((np.arange(100).reshape(10, 10))[2:, 2:]).reshape(1, 1, 8, 8)
    assert np.allclose(result_arr, result_arr_ref)

    # test 2d with strides
    strides = [2, 2]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ng.max_pool(A, strides, pads_begin, pads_end, window_shape)
    function = Function([model], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    size = 4
    result = backend.create_tensor(element_type, Shape([1, 1, size, size]))
    result_arr = np.zeros(size * size, dtype=np.float32).reshape(1, 1, size, size)

    result.write(util.numpy_to_c(result_arr), size * size * 4)
    handle = backend.compile(function)
    handle.call([result], [a])
    result.read(util.numpy_to_c(result_arr), size * size * 4)

    result_arr_ref = ((np.arange(100).reshape(10, 10))[2::2, 2::2]).reshape(1, 1, size, size)
    assert np.allclose(result_arr, result_arr_ref)


def convolution2d(
    image,
    filterit,
    strides=(1, 1),
    dilation=(1, 1),
    padding_below=(0, 0),
    padding_above=(0, 0),
    data_dilation=(1, 1),
):
    def dilate(arr, dil=(1, 1)):
        m, n = arr.shape
        new_m, new_n = (m - 1) * dil[0] + 1, (n - 1) * dil[1] + 1
        new_arr = np.zeros(new_m * new_n, dtype=np.float32).reshape(new_m, new_n)
        for i in range(m):
            for j in range(n):
                new_arr[dil[0] * i][dil[1] * j] = arr[i][j]
        return new_arr

    i_m, i_n = image.shape
    new_image = np.zeros(
        (i_m + padding_below[0] + padding_above[0]) * (i_n + padding_below[1] + padding_above[1]),
        dtype=np.float32,
    ).reshape(i_m + padding_below[0] + padding_above[0], i_n + padding_below[1] + padding_above[1])
    new_image[
        padding_below[0] : padding_below[0] + i_m, padding_below[1] : padding_below[1] + i_n
    ] = image
    image = new_image
    image = image if data_dilation[0] == data_dilation[1] == 1 else dilate(image, data_dilation)
    i_m, i_n = image.shape

    filterit = filterit if dilation[0] == dilation[1] == 1 else dilate(filterit, dilation)
    f_m, f_n = filterit.shape

    # result_shape
    r_m = i_m - f_m + 1
    r_n = i_n - f_n + 1
    r_m //= strides[0]
    r_n //= strides[1]

    result = np.zeros(r_m * r_n, dtype=np.float32).reshape(r_m, r_n)

    for i in range(r_m):
        for j in range(r_n):
            sub_m = image[
                i * strides[0] : i * strides[0] + f_m, j * strides[1] : j * strides[1] + f_n
            ]
            result[i][j] = np.sum(sub_m * filterit)
    return result


def test_convolution_simple():

    element_type = Type.f32
    image_shape = Shape([1, 1, 16, 16])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    parameter_list = [data, filters]

    image_arr = np.arange(-128, 128, 1, dtype=np.float32).reshape(1, 1, 16, 16)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][0][0] = -1
    filter_arr[0][0][1][1] = -1
    filter_arr[0][0][2][2] = -1
    filter_arr[0][0][0][2] = -1
    filter_arr[0][0][2][0] = -1
    result_arr = np.zeros(196, dtype=np.float32).reshape(1, 1, 14, 14)

    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [1, 1]

    model = ng.convolution(data, filters, strides, pads_begin, pads_end, dilations)
    function = Function([model], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 16 * 16 * 4)
    b.write(util.numpy_to_c(filter_arr), 3 * 3 * 4)

    result = backend.create_tensor(element_type, Shape([1, 1, 14, 14]))
    result.write(util.numpy_to_c(result_arr), 14 * 14 * 4)
    handle = backend.compile(function)
    handle.call([result], [a, b])
    result.read(util.numpy_to_c(result_arr), 14 * 14 * 4)

    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0]).reshape(1, 1, 14, 14)
    assert np.allclose(result_arr, result_arr_ref)


def test_convolution_with_strides():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    parameter_list = [data, filters]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.zeros(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][1][1] = 1
    strides = [2, 2]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [1, 1]

    model = ng.convolution(data, filters, strides, pads_begin, pads_end, dilations)
    function = Function([model], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 10 * 10 * 4)
    b.write(util.numpy_to_c(filter_arr), 3 * 3 * 4)

    result_arr = np.zeros(16, dtype=np.float32).reshape(1, 1, 4, 4)
    result = backend.create_tensor(element_type, Shape([1, 1, 4, 4]))
    result.write(util.numpy_to_c(result_arr), 4 * 4 * 4)
    handle = backend.compile(function)
    handle.call([result], [a, b])

    result.read(util.numpy_to_c(result_arr), 4 * 4 * 4)
    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0], strides).reshape(1, 1, 4, 4)
    assert np.allclose(result_arr, result_arr_ref)


def test_convolution_with_filter_dilation():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    parameter_list = [data, filters]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [2, 2]

    model = ng.convolution(data, filters, strides, pads_begin, pads_end, dilations)
    function = Function([model], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 10 * 10 * 4)
    b.write(util.numpy_to_c(filter_arr), 3 * 3 * 4)

    result_arr = np.zeros(36, dtype=np.float32).reshape(1, 1, 6, 6)
    result = backend.create_tensor(element_type, Shape([1, 1, 6, 6]))
    result.write(util.numpy_to_c(result_arr), 6 * 6 * 4)
    handle = backend.compile(function)
    handle.call([result], [a, b])

    result.read(util.numpy_to_c(result_arr), 6 * 6 * 4)
    result_arr_ref = convolution2d(image_arr[0][0], filter_arr[0][0], strides, dilations).reshape(
        1, 1, 6, 6
    )
    assert np.allclose(result_arr, result_arr_ref)


def test_convolution_with_padding():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    parameter_list = [data, filters]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = np.zeros(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][1][1] = 1
    strides = [1, 1]
    dilations = [2, 2]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ng.convolution(data, filters, strides, pads_begin, pads_end, dilations)
    function = Function([model], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 10 * 10 * 4)
    b.write(util.numpy_to_c(filter_arr), 3 * 3 * 4)

    result_arr = np.zeros(36, dtype=np.float32).reshape(1, 1, 6, 6)
    result = backend.create_tensor(element_type, Shape([1, 1, 6, 6]))
    result.write(util.numpy_to_c(result_arr), 6 * 6 * 4)
    handle = backend.compile(function)
    handle.call([result], [a, b])

    result.read(util.numpy_to_c(result_arr), 6 * 6 * 4)
    result_arr_ref = convolution2d(
        image_arr[0][0], filter_arr[0][0], strides, dilations, pads_begin, pads_end
    ).reshape(1, 1, 6, 6)
    assert np.allclose(result_arr, result_arr_ref)


def test_convolution_with_non_zero_padding():
    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    parameter_list = [data, filters]

    image_arr = np.arange(100, dtype=np.float32).reshape(1, 1, 10, 10)
    filter_arr = (np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)) * -1
    filter_arr[0][0][1][1] = 1
    strides = [1, 1]
    dilations = [2, 2]
    pads_begin = [2, 1]
    pads_end = [1, 2]

    model = ng.convolution(data, filters, strides, pads_begin, pads_end, dilations)
    function = Function([model], parameter_list, "test")
    backend = Backend.create(test.BACKEND_NAME)

    a = backend.create_tensor(element_type, image_shape)
    b = backend.create_tensor(element_type, filter_shape)

    a.write(util.numpy_to_c(image_arr), 10 * 10 * 4)
    b.write(util.numpy_to_c(filter_arr), 3 * 3 * 4)

    result_arr = np.zeros(81, dtype=np.float32).reshape(1, 1, 9, 9)
    result = backend.create_tensor(element_type, Shape([1, 1, 9, 9]))
    result.write(util.numpy_to_c(result_arr), 9 * 9 * 4)
    handle = backend.compile(function)
    handle.call([result], [a, b])

    result.read(util.numpy_to_c(result_arr), 9 * 9 * 4)
    result_arr_ref = convolution2d(
        image_arr[0][0], filter_arr[0][0], strides, dilations, pads_begin, pads_end
    ).reshape(1, 1, 9, 9)
    assert np.allclose(result_arr, result_arr_ref)
