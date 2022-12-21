# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import numpy as np

import openvino.runtime.opset8 as ov
from openvino.runtime import AxisSet, Shape, Type
from openvino.runtime.op import Constant, Parameter


def binary_op(op_str, a, b):

    if op_str == "+":
        return a + b
    elif op_str == "Add":
        return ov.add(a, b)
    elif op_str == "-":
        return a - b
    elif op_str == "Sub":
        return ov.subtract(a, b)
    elif op_str == "*":
        return a * b
    elif op_str == "Mul":
        return ov.multiply(a, b)
    elif op_str == "/":
        return a / b
    elif op_str == "Div":
        return ov.divide(a, b)
    elif op_str == "Equal":
        return ov.equal(a, b)
    elif op_str == "Greater":
        return ov.greater(a, b)
    elif op_str == "GreaterEq":
        return ov.greater_equal(a, b)
    elif op_str == "Less":
        return ov.less(a, b)
    elif op_str == "LessEq":
        return ov.less_equal(a, b)
    elif op_str == "Maximum":
        return ov.maximum(a, b)
    elif op_str == "Minimum":
        return ov.minimum(a, b)
    elif op_str == "NotEqual":
        return ov.not_equal(a, b)
    elif op_str == "Power":
        return ov.power(a, b)


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


def binary_op_exec(op_str, expected_ov_str=None):
    if not expected_ov_str:
        expected_ov_str = op_str

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    node = binary_op(op_str, A, B)

    assert node.get_type_name() == expected_ov_str
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]
    assert node.get_output_element_type(0) == Type.f32


def binary_op_comparison(op_str, expected_ov_str=None):
    if not expected_ov_str:
        expected_ov_str = op_str

    element_type = Type.f32
    shape = Shape([2, 2])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    node = binary_op(op_str, A, B)

    assert node.get_type_name() == expected_ov_str
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]
    assert node.get_output_element_type(0) == Type.boolean


def test_add():
    binary_op_exec("+", "Add")


def test_add_op():
    binary_op_exec("Add")


def test_sub():
    binary_op_exec("-", "Subtract")


def test_sub_op():
    binary_op_exec("Sub", "Subtract")


def test_mul():
    binary_op_exec("*", "Multiply")


def test_mul_op():
    binary_op_exec("Mul", "Multiply")


def test_div():
    binary_op_exec("/", "Divide")


def test_div_op():
    binary_op_exec("Div", "Divide")


def test_maximum():
    binary_op_exec("Maximum")


def test_minimum():
    binary_op_exec("Minimum")


def test_power():
    binary_op_exec("Power")


def test_greater():
    binary_op_comparison("Greater")


def test_greater_eq():
    binary_op_comparison("GreaterEq", "GreaterEqual")


def test_less():
    binary_op_comparison("Less")


def test_less_eq():
    binary_op_comparison("LessEq", "LessEqual")


def test_not_equal():
    binary_op_comparison("NotEqual")


def test_add_with_mul():

    element_type = Type.f32
    shape = Shape([4])
    A = Parameter(element_type, shape)
    B = Parameter(element_type, shape)
    C = Parameter(element_type, shape)
    node = ov.multiply(ov.add(A, B), C)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [4]
    assert node.get_output_element_type(0) == Type.f32


def unary_op(op_str, a):
    if op_str == "Abs":
        return ov.abs(a)
    elif op_str == "Acos":
        return ov.acos(a)
    elif op_str == "Acosh":
        return ov.acosh(a)
    elif op_str == "Asin":
        return ov.asin(a)
    elif op_str == "Asinh":
        return ov.asinh(a)
    elif op_str == "Atan":
        return ov.atan(a)
    elif op_str == "Atanh":
        return ov.atanh(a)
    elif op_str == "Ceiling":
        return ov.ceiling(a)
    elif op_str == "Cos":
        return ov.cos(a)
    elif op_str == "Cosh":
        return ov.cosh(a)
    elif op_str == "Floor":
        return ov.floor(a)
    elif op_str == "log":
        return ov.log(a)
    elif op_str == "exp":
        return ov.exp(a)
    elif op_str == "negative":
        return ov.negative(a)
    elif op_str == "Sign":
        return ov.sign(a)
    elif op_str == "Sin":
        return ov.sin(a)
    elif op_str == "Sinh":
        return ov.sinh(a)
    elif op_str == "Sqrt":
        return ov.sqrt(a)
    elif op_str == "Tan":
        return ov.tan(a)
    elif op_str == "Tanh":
        return ov.tanh(a)


def unary_op_ref(op_str, a):
    if op_str == "Abs":
        return np.abs(a)
    elif op_str == "Acos":
        return np.arccos(a)
    elif op_str == "Acosh":
        return np.arccosh(a)
    elif op_str == "Asin":
        return np.arcsin(a)
    elif op_str == "Asinh":
        return np.arcsinh(a)
    elif op_str == "Atan":
        return np.arctan(a)
    elif op_str == "Atanh":
        return np.arctanh(a)
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


def unary_op_exec(op_str, input_list, expected_ov_str=None):
    """
    input_list needs to have deep length of 4
    """
    if not expected_ov_str:
        expected_ov_str = op_str

    element_type = Type.f32
    shape = Shape(np.array(input_list).shape)
    A = Parameter(element_type, shape)
    node = unary_op(op_str, A)

    assert node.get_type_name() == expected_ov_str
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(shape)
    assert node.get_output_element_type(0) == Type.f32


def test_abs():
    input_list = [-1, 0, 1, 2]
    op_str = "Abs"
    unary_op_exec(op_str, input_list)


def test_acos():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Acos"
    unary_op_exec(op_str, input_list)


def test_acosh():
    input_list = [2., 3., 1.5, 1.0]
    op_str = "Acosh"
    unary_op_exec(op_str, input_list)


def test_asin():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Asin"
    unary_op_exec(op_str, input_list)


def test_asinh():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Asinh"
    unary_op_exec(op_str, input_list)


def test_atan():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Atan"
    unary_op_exec(op_str, input_list)


def test_atanh():
    input_list = [-1, 0, 0.5, 1]
    op_str = "Atanh"
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
    unary_op_exec(op_str, input_list, "Log")


def test_exp():
    input_list = [-1, 0, 1, 2]
    op_str = "exp"
    unary_op_exec(op_str, input_list, "Exp")


def test_negative():
    input_list = [-1, 0, 1, 2]
    op_str = "negative"
    unary_op_exec(op_str, input_list, "Negative")


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


def test_reshape():
    element_type = Type.f32
    shape = Shape([2, 3])
    A = Parameter(element_type, shape)
    node = ov.reshape(A, Shape([3, 2]), special_zero=False)

    assert node.get_type_name() == "Reshape"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == element_type


def test_broadcast():
    element_type = Type.f32
    A = Parameter(element_type, Shape([3]))
    node = ov.broadcast(A, [3, 3])
    assert node.get_type_name() == "Broadcast"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 3]
    assert node.get_output_element_type(0) == element_type


def test_constant():
    element_type = Type.f32
    node = Constant(element_type, Shape([3, 3]), list(range(9)))
    assert node.get_type_name() == "Constant"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 3]
    assert node.get_output_element_type(0) == element_type


def test_constant_opset_ov_type():
    node = ov.constant(np.arange(9).reshape(3, 3), Type.f32)
    assert node.get_type_name() == "Constant"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 3]
    assert node.get_output_element_type(0) == Type.f32


def test_constant_opset_numpy_type():
    node = ov.constant(np.arange(9).reshape(3, 3), np.float32)
    assert node.get_type_name() == "Constant"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 3]
    assert node.get_output_element_type(0) == Type.f32


def test_concat():
    element_type = Type.f32
    A = Parameter(element_type, Shape([1, 2]))
    B = Parameter(element_type, Shape([1, 2]))
    C = Parameter(element_type, Shape([1, 2]))
    node = ov.concat([A, B, C], axis=0)
    assert node.get_type_name() == "Concat"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == element_type


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
    node = ov.select(A, B, C)
    assert node.get_type_name() == "Select"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 2]
    assert node.get_output_element_type(0) == element_type


def test_max_pool_1d():
    element_type = Type.f32
    shape = Shape([1, 1, 10])
    A = Parameter(element_type, shape)
    window_shape = [3]

    strides = [1] * len(window_shape)
    dilations = [1] * len(window_shape)
    pads_begin = [0] * len(window_shape)
    pads_end = [0] * len(window_shape)
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    model = ov.max_pool(
        A,
        strides,
        dilations,
        pads_begin,
        pads_end,
        window_shape,
        rounding_type,
        auto_pad,
        idx_elem_type,
    )
    assert model.get_type_name() == "MaxPool"
    assert model.get_output_size() == 2
    assert list(model.get_output_shape(0)) == [1, 1, 8]
    assert list(model.get_output_shape(1)) == [1, 1, 8]
    assert model.get_output_element_type(0) == element_type
    assert model.get_output_element_type(1) == Type.i32

def test_max_pool_1d_with_strides():
    element_type = Type.f32
    shape = Shape([1, 1, 10])
    A = Parameter(element_type, shape)
    window_shape = [3]
    strides = [2]
    pads_begin = [0] * len(window_shape)
    dilations = [1] * len(window_shape)
    pads_end = [0] * len(window_shape)
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    model = ov.max_pool(
        A,
        strides,
        dilations,
        pads_begin,
        pads_end,
        window_shape,
        rounding_type,
        auto_pad,
        idx_elem_type,
    )

    assert model.get_type_name() == "MaxPool"
    assert model.get_output_size() == 2
    assert list(model.get_output_shape(0)) == [1, 1, 4]
    assert list(model.get_output_shape(1)) == [1, 1, 4]
    assert model.get_output_element_type(0) == element_type
    assert model.get_output_element_type(1) == Type.i32

def test_max_pool_2d():
    element_type = Type.f32
    shape = Shape([1, 1, 10, 10])
    A = Parameter(element_type, shape)
    window_shape = [3, 3]
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ov.max_pool(
        A,
        strides,
        dilations,
        pads_begin,
        pads_end,
        window_shape,
        rounding_type,
        auto_pad,
        idx_elem_type,
    )
    assert model.get_type_name() == "MaxPool"
    assert model.get_output_size() == 2
    assert list(model.get_output_shape(0)) == [1, 1, 8, 8]
    assert list(model.get_output_shape(1)) == [1, 1, 8, 8]
    assert model.get_output_element_type(0) == element_type
    assert model.get_output_element_type(1) == Type.i32


def test_max_pool_2d_with_strides():
    element_type = Type.f32
    shape = Shape([1, 1, 10, 10])
    A = Parameter(element_type, shape)
    strides = [2, 2]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    window_shape = [3, 3]
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    model = ov.max_pool(
        A,
        strides,
        dilations,
        pads_begin,
        pads_end,
        window_shape,
        rounding_type,
        auto_pad,
        idx_elem_type,
    )
    assert model.get_type_name() == "MaxPool"
    assert model.get_output_size() == 2
    assert list(model.get_output_shape(0)) == [1, 1, 4, 4]
    assert list(model.get_output_shape(1)) == [1, 1, 4, 4]
    assert model.get_output_element_type(0) == element_type
    assert model.get_output_element_type(1) == Type.i32


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
    new_image[padding_below[0] : padding_below[0] + i_m, padding_below[1] : padding_below[1] + i_n] = image
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
            sub_m = image[i * strides[0] : i * strides[0] + f_m, j * strides[1] : j * strides[1] + f_n]
            result[i][j] = np.sum(sub_m * filterit)
    return result


def test_convolution_simple():
    element_type = Type.f32
    image_shape = Shape([1, 1, 16, 16])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    filter_arr = np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][0][0] = -1
    filter_arr[0][0][1][1] = -1
    filter_arr[0][0][2][2] = -1
    filter_arr[0][0][0][2] = -1
    filter_arr[0][0][2][0] = -1

    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [1, 1]

    model = ov.convolution(data, filters, strides, pads_begin, pads_end, dilations)

    assert model.get_type_name() == "Convolution"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 1, 14, 14]
    assert model.get_output_element_type(0) == element_type


def test_convolution_with_strides():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    filter_arr = np.zeros(9, dtype=np.float32).reshape([1, 1, 3, 3])
    filter_arr[0][0][1][1] = 1
    strides = [2, 2]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [1, 1]

    model = ov.convolution(data, filters, strides, pads_begin, pads_end, dilations)

    assert model.get_type_name() == "Convolution"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 1, 4, 4]
    assert model.get_output_element_type(0) == element_type


def test_convolution_with_filter_dilation():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    strides = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    dilations = [2, 2]

    model = ov.convolution(data, filters, strides, pads_begin, pads_end, dilations)

    assert model.get_type_name() == "Convolution"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 1, 6, 6]
    assert model.get_output_element_type(0) == element_type


def test_convolution_with_padding():

    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    filter_arr = np.zeros(9, dtype=np.float32).reshape(1, 1, 3, 3)
    filter_arr[0][0][1][1] = 1
    strides = [1, 1]
    dilations = [2, 2]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ov.convolution(data, filters, strides, pads_begin, pads_end, dilations)

    assert model.get_type_name() == "Convolution"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 1, 6, 6]
    assert model.get_output_element_type(0) == element_type


def test_convolution_with_non_zero_padding():
    element_type = Type.f32
    image_shape = Shape([1, 1, 10, 10])
    filter_shape = Shape([1, 1, 3, 3])
    data = Parameter(element_type, image_shape)
    filters = Parameter(element_type, filter_shape)
    filter_arr = (np.ones(9, dtype=np.float32).reshape(1, 1, 3, 3)) * -1
    filter_arr[0][0][1][1] = 1
    strides = [1, 1]
    dilations = [2, 2]
    pads_begin = [2, 1]
    pads_end = [1, 2]

    model = ov.convolution(data, filters, strides, pads_begin, pads_end, dilations)

    assert model.get_type_name() == "Convolution"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [1, 1, 9, 9]
    assert model.get_output_element_type(0) == element_type
