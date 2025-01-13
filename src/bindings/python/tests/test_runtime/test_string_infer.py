# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset13 as ops
from openvino import (
    CompiledModel,
    InferRequest,
    Model,
    Type,
    compile_model,
)


def create_string_compiled_model(shape):
    parameter = ops.parameter(shape, Type.string)
    result = ops.result(parameter)

    model = Model([result], [parameter])
    compiled = compile_model(model)
    return compiled


def create_string_infer_request(shape):
    return create_string_compiled_model(shape).create_infer_request()


def as_bytes_array(data):
    array = np.array(data)
    return array if array.dtype.kind == "S" else np.char.encode(array)


def as_string_array(data):
    array = np.array(data)
    return array if array.dtype.kind == "U" else np.char.decode(array)


@pytest.mark.parametrize(
    ("ov_func"),
    [
        create_string_compiled_model([-1]).__call__,
        create_string_infer_request([-1]).infer,
    ],
)
def test_keyword_only_decode_fails(ov_func):
    with pytest.raises(TypeError) as error:
        _ = ov_func([], False, False, False)
    assert "takes from 1 to 4 positional arguments but 5 were given" in str(error.value)


@pytest.mark.parametrize(
    ("class_defaults", "expected_value"),
    [
        (CompiledModel.__call__.__kwdefaults__, True),
        (InferRequest.infer.__kwdefaults__, True),
    ],
)
def test_default_decode_flag(class_defaults, expected_value):
    assert class_defaults["decode_strings"] == expected_value


@pytest.mark.parametrize(
    ("string_data", "data_shape"),
    [
        ([bytes("text", encoding="utf-8"), bytes("openvino", encoding="utf-8")], [-1]),
        ([[b"xyz"], [b"abc"], [b"this is my last"]], [3, -1]),
        ([b"text\0with\0null", b"openvino\0"], [-1]),
        (["text", "abc", "openvino"], [3]),
        (["text", "больше текста", "jeszcze więcej słów", "효과가 있었어"], [-1]),
        ([["text"], ["abc"], ["openvino"]], [3, 1]),
        ([["jeszcze więcej słów", "효과가 있었어"]], [1, 2]),
    ],
)
@pytest.mark.parametrize(
    ("decode_strings"),
    [
        True,
        False,
    ],
)
def test_infer_request_infer(string_data, data_shape, decode_strings):
    infer_request = create_string_infer_request(data_shape)
    res = infer_request.infer(string_data, decode_strings=decode_strings)
    assert np.array_equal(res[0], as_string_array(string_data) if decode_strings else as_bytes_array(string_data))


@pytest.mark.parametrize(
    ("string_data", "data_shape"),
    [
        ([bytes("text", encoding="utf-8"), bytes("openvino", encoding="utf-8")], [-1]),
        ([[b"xyz"], [b"abc"], [b"this is my last"]], [3, -1]),
        ([b"text\0with\0null", b"openvino\0"], [-1]),
        (["text", "abc", "openvino"], [3]),
        (["text", "больше текста", "jeszcze więcej słów", "효과가 있었어"], [-1]),
        ([["text"], ["abc"], ["openvino"]], [3, 1]),
        ([["jeszcze więcej słów", "효과가 있었어"]], [1, 2]),
    ],
)
@pytest.mark.parametrize(
    ("decode_strings"),
    [
        True,
        False,
    ],
)
def test_compiled_model_infer(string_data, data_shape, decode_strings):
    compiled_model = create_string_compiled_model(data_shape)
    res = compiled_model(string_data, decode_strings=decode_strings)
    assert np.array_equal(res[0], as_string_array(string_data) if decode_strings else as_bytes_array(string_data))
