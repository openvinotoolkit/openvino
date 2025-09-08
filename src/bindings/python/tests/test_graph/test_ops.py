# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import numpy as np
import pytest
from contextlib import nullcontext as does_not_raise

import openvino.opset8 as ov
from openvino import Shape, Type
from openvino import AxisSet
from openvino.op import Constant, Parameter


@pytest.mark.parametrize(("ov_op", "expected_ov_str", "expected_type"), [
    (lambda a, b: a + b, "Add", Type.f32),
    (ov.add, "Add", Type.f32),
    (lambda a, b: a - b, "Subtract", Type.f32),
    (ov.subtract, "Subtract", Type.f32),
    (lambda a, b: a * b, "Multiply", Type.f32),
    (ov.multiply, "Multiply", Type.f32),
    (lambda a, b: a / b, "Divide", Type.f32),
    (ov.divide, "Divide", Type.f32),
    (ov.maximum, "Maximum", Type.f32),
    (ov.minimum, "Minimum", Type.f32),
    (ov.power, "Power", Type.f32),
    (ov.equal, "Equal", Type.boolean),
    (ov.greater, "Greater", Type.boolean),
    (ov.greater_equal, "GreaterEqual", Type.boolean),
    (ov.less, "Less", Type.boolean),
    (ov.less_equal, "LessEqual", Type.boolean),
    (ov.not_equal, "NotEqual", Type.boolean),
])
def test_binary_op(ov_op, expected_ov_str, expected_type):
    element_type = Type.f32
    shape = Shape([2, 2])
    param1 = Parameter(element_type, shape)
    param2 = Parameter(element_type, shape)
    node = ov_op(param1, param2)

    assert node.get_type_name() == expected_ov_str
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]
    assert node.get_output_element_type(0) == expected_type


def test_add_with_mul():

    element_type = Type.f32
    shape = Shape([4])
    param1 = Parameter(element_type, shape)
    param2 = Parameter(element_type, shape)
    param3 = Parameter(element_type, shape)
    node = ov.multiply(ov.add(param1, param2), param3)

    assert node.get_type_name() == "Multiply"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [4]
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize(("ov_op", "expected_ov_str"), [
    (ov.abs, "Abs"),
    (ov.acos, "Acos"),
    (ov.acosh, "Acosh"),
    (ov.asin, "Asin"),
    (ov.asinh, "Asinh"),
    (ov.atan, "Atan"),
    (ov.atanh, "Atanh"),
    (ov.ceiling, "Ceiling"),
    (ov.cos, "Cos"),
    (ov.cosh, "Cosh"),
    (ov.floor, "Floor"),
    (ov.log, "Log"),
    (ov.exp, "Exp"),
    (ov.negative, "Negative"),
    (ov.sign, "Sign"),
    (ov.sin, "Sin"),
    (ov.sinh, "Sinh"),
    (ov.sqrt, "Sqrt"),
    (ov.tan, "Tan"),
    (ov.tanh, "Tanh"),
])
def test_unary_op(ov_op, expected_ov_str):

    element_type = Type.f32
    shape = Shape([4])
    param1 = Parameter(element_type, shape)
    node = ov_op(param1)

    assert node.get_type_name() == expected_ov_str
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(shape)
    assert node.get_output_element_type(0) == Type.f32


def test_reshape():
    element_type = Type.f32
    shape = Shape([2, 3])
    param1 = Parameter(element_type, shape)
    node = ov.reshape(param1, Shape([3, 2]), special_zero=False)

    assert node.get_type_name() == "Reshape"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == element_type


def test_broadcast():
    element_type = Type.f32
    param1 = Parameter(element_type, Shape([3]))
    node = ov.broadcast(param1, [3, 3])
    assert node.get_type_name() == "Broadcast"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 3]
    assert node.get_output_element_type(0) == element_type


@pytest.mark.parametrize(("const", "args", "expectation"), [
    (Constant, (Type.f32, Shape([3, 3]), list(range(9))), does_not_raise()),
    (ov.constant, (np.arange(9).reshape(3, 3), Type.f32), does_not_raise()),
    (ov.constant, (np.arange(9).reshape(3, 3), np.float32), does_not_raise()),
    (ov.constant, [None], pytest.raises(ValueError)),
])
def test_constant(const, args, expectation):
    with expectation:
        node = const(*args)
        assert node.get_type_name() == "Constant"
        assert node.get_output_size() == 1
        assert list(node.get_output_shape(0)) == [3, 3]
        assert node.get_output_element_type(0) == Type.f32
        assert node.get_byte_size() == 36


def test_concat():
    element_type = Type.f32
    param1 = Parameter(element_type, Shape([1, 2]))
    param2 = Parameter(element_type, Shape([1, 2]))
    param3 = Parameter(element_type, Shape([1, 2]))
    node = ov.concat([param1, param2, param3], axis=0)
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
    param1 = Parameter(Type.boolean, Shape([1, 2]))
    param2 = Parameter(element_type, Shape([1, 2]))
    param3 = Parameter(element_type, Shape([1, 2]))
    node = ov.select(param1, param2, param3)
    assert node.get_type_name() == "Select"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 2]
    assert node.get_output_element_type(0) == element_type


def test_max_pool_1d():
    element_type = Type.f32
    shape = Shape([1, 1, 10])
    param1 = Parameter(element_type, shape)
    window_shape = [3]

    strides = [1] * len(window_shape)
    dilations = [1] * len(window_shape)
    pads_begin = [0] * len(window_shape)
    pads_end = [0] * len(window_shape)
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    model = ov.max_pool(
        param1,
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
    param1 = Parameter(element_type, shape)
    window_shape = [3]
    strides = [2]
    pads_begin = [0] * len(window_shape)
    dilations = [1] * len(window_shape)
    pads_end = [0] * len(window_shape)
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    model = ov.max_pool(
        param1,
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
    param1 = Parameter(element_type, shape)
    window_shape = [3, 3]
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    strides = [1, 1]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]

    model = ov.max_pool(
        param1,
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
    param1 = Parameter(element_type, shape)
    strides = [2, 2]
    dilations = [1, 1]
    pads_begin = [0, 0]
    pads_end = [0, 0]
    window_shape = [3, 3]
    rounding_type = "floor"
    auto_pad = "explicit"
    idx_elem_type = "i32"

    model = ov.max_pool(
        param1,
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
