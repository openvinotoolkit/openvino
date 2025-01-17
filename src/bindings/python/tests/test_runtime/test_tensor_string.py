# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov

import pytest

from enum import Enum


class DataGetter(Enum):
    BYTES = 1
    STRINGS = 2


def _check_tensor_string(tensor_data, test_data):
    assert tensor_data.shape == test_data.shape
    assert tensor_data.strides == test_data.strides
    assert np.array_equal(tensor_data, test_data)
    assert not (np.shares_memory(tensor_data, test_data))


def check_bytes_based(tensor, string_data, to_flat=False):
    tensor_data = tensor.bytes_data
    encoded_data = string_data if string_data.dtype.kind == "S" else np.char.encode(string_data)
    assert tensor_data.dtype.kind == "S"
    _check_tensor_string(tensor_data.flatten() if to_flat else tensor_data, encoded_data.flatten() if to_flat else encoded_data)


def check_string_based(tensor, string_data, to_flat=False):
    tensor_data = tensor.str_data
    decoded_data = string_data if string_data.dtype.kind == "U" else np.char.decode(string_data)
    assert tensor_data.dtype.kind == "U"
    _check_tensor_string(tensor_data.flatten() if to_flat else tensor_data, decoded_data.flatten() if to_flat else decoded_data)


def test_string_tensor_shared_memory_fails():
    data = np.array(["You", "shall", "not", "pass!"])
    with pytest.raises(RuntimeError) as e:
        _ = ov.Tensor(data, shared_memory=True)
    assert "SHARED MEMORY MODE FOR THIS TENSOR IS NOT APPLICABLE! String types can be only copied." in str(e.value)


def test_string_tensor_data_warning():
    data = np.array(["You", "shall", "not", "pass!"])
    tensor = ov.Tensor(data, shared_memory=False)
    with pytest.warns(RuntimeWarning) as w:
        _ = tensor.data
    assert "Data of string type will be copied! Please use dedicated properties" in str(w[0].message)


@pytest.mark.parametrize(
    ("init_type"),
    [
        (ov.Type.string),
        (str),
        (bytes),
        (np.str_),
        (np.bytes_),
    ],
)
def test_empty_string_tensor(init_type):
    tensor = ov.Tensor(type=init_type, shape=ov.Shape([2, 2]))
    assert tensor.element_type == ov.Type.string


@pytest.mark.parametrize(
    ("string_data"),
    [
        ([bytes("text", encoding="utf-8"), bytes("openvino", encoding="utf-8")]),
        ([[b"xyz"], [b"abc"], [b"this is my last"]]),
        ([[b"text\0with\0null"], [b"openvino\0"]]),
        (["text", "abc", "openvino"]),
        (["text", "больше текста", "jeszcze więcej słów", "효과가 있었어"]),
        ([["text"], ["abc"], ["openvino"]]),
        ([["text"]]),
        (["tex\u0000t\u0000tt"]),
        ([["abĆ"]]),
        ([["tex\u0000tttt"], ["abĆ"]]),
        ([["jeszcze więcej słówe"], [u"효#과가 있었어"]]),
        ([["jeszcze\u0000 więcej słówekó"]]),
        ([["효과가 있었어"]]),
        (["ab\u0000Ć"]),
    ])
def test_init_with_list(string_data):
    tensor = ov.Tensor(string_data)
    assert tensor.element_type == ov.Type.string
    # Convert to numpy to perform all checks. Memory is not shared,
    # so it does not matter if data is stored in numpy format.
    _string_data = np.array(string_data)
    # Encoded:
    check_bytes_based(tensor, _string_data)
    # Decoded:
    check_string_based(tensor, _string_data)


def test_init_with_list_rare_real_scenario():
    input_data = ["tex\u0000\u0000ttt\u0000\u0000", "ab\u0000Ć"]
    tensor = ov.Tensor(input_data)
    assert tensor.element_type == ov.Type.string
    # Convert to numpy to perform all checks. Memory is not shared,
    np_string_data = np.array(input_data)
    # Encoded:
    check_bytes_based(tensor, np_string_data)
    # Decoded:
    str_tensor_data = tensor.str_data
    assert str_tensor_data.shape == np_string_data.shape
    # case when OV is not aligned with numpy format
    # strides are different as trailing null characters are not stored in the tensor
    # is rare to have any use of trailing null character in the string
    assert str_tensor_data.strides != np_string_data.strides
    assert np.array_equal(str_tensor_data, np_string_data)
    assert not (np.shares_memory(str_tensor_data, np_string_data))


@pytest.mark.parametrize(
    ("string_data"),
    [
        (np.array(["text", "abc", "openvino"]).astype("S")),  # "|S"
        (np.array([["xyz"], ["abc"]]).astype(np.bytes_)),  # "|S"
        (np.array(["text", "abc", "openvino"])),  # "<U"
        (np.array(["text", "больше текста", "jeszcze więcej słów", "효과가 있었어"])),  # "<U"
        (np.array([["text"], ["abc"], ["openvino"]])),  # "<U"
        (np.array([["jeszcze więcej słów", "효과가 있었어"]])),  # "<U"
    ],
)
def test_init_with_numpy(string_data):
    tensor = ov.Tensor(string_data, shared_memory=False)
    assert tensor.element_type == ov.Type.string
    # Encoded:
    check_bytes_based(tensor, string_data)
    # Decoded:
    check_string_based(tensor, string_data)


@pytest.mark.parametrize(
    ("init_type"),
    [
        (ov.Type.string),
        (str),
        (bytes),
        (np.str_),
        (np.bytes_),
    ],
)
@pytest.mark.parametrize(
    ("init_shape"),
    [
        (ov.Shape()),
        (ov.Shape([])),
        (ov.Shape([5])),
        (ov.Shape([1, 1])),
        (ov.Shape([2, 4, 5])),
    ],
)
@pytest.mark.parametrize(
    ("string_data"),
    [
        (np.array(["text", "abc", "openvino"]).astype(np.bytes_)),  # "|S8"
        (np.array([["text!"], ["abc?"]]).astype("S")),  # "|S8"
        (np.array(["text", "abc", "openvino"])),  # "<U", depending on platform
        (np.array([["text"], ["abc"], ["openvino"]])),  # "<U", depending on platform
        (np.array([["text", "больше текста"], ["jeszcze więcej słów", "효과가 있었어"]])),  # "<U"
        (np.array([["#text@", "больше текста"]])),  # "<U"
    ],
)
def test_empty_tensor_copy_from(init_type, init_shape, string_data):
    tensor = ov.Tensor(init_type, init_shape)
    assert tensor.element_type == ov.Type.string
    tensor.copy_from(string_data)
    # Encoded:
    check_bytes_based(tensor, string_data)
    # Decoded:
    check_string_based(tensor, string_data)


@pytest.mark.parametrize(
    ("init_shape"),
    [
        (ov.Shape()),
        (ov.Shape([])),
        (ov.Shape([1])),
        (ov.Shape([8])),
        (ov.Shape([4, 4])),
    ],
)
@pytest.mark.parametrize(
    ("string_data"),
    [
        (np.array(["text", "abc", "openvino"]).astype(np.bytes_)),  # "|S"
        (np.array([["text!"], ["abc?"]]).astype("S")),  # "|S"
        (np.array(["text", "abc", "openvino"])),  # "<U"
        (np.array([["text"], ["abc"], ["openvino"]])),  # "<U"
        (np.array([["text", "больше текста"], ["jeszcze więcej słów", "효과가 있었어"]])),  # "<U"
        (np.array([["#text@", "больше текста"]])),  # "<U"
        ([bytes("text", encoding="utf-8"), bytes("openvino", encoding="utf-8")]),
        ([[b"xyz"], [b"abc"], [b"this is my last"]]),
        (["text", "abc", "openvino"]),
        (["text", "больше текста", "jeszcze więcej słów", "효과가 있었어"]),
        ([["text"], ["abc"], ["openvino"]]),
        ([["jeszcze więcej słów", "효과가 있었어"]]),
    ],
)
def test_populate_fails_size_check(init_shape, string_data):
    tensor = ov.Tensor(ov.Type.string, init_shape)
    assert tensor.element_type == ov.Type.string
    with pytest.raises(RuntimeError) as e:
        tensor.bytes_data = string_data
    assert "Passed array must have the same size (number of elements) as the Tensor!" in str(e.value)
    with pytest.raises(RuntimeError) as e:
        tensor.str_data = string_data
    assert "Passed array must have the same size (number of elements) as the Tensor!" in str(e.value)


@pytest.mark.parametrize(
    ("string_data"),
    [
        (np.array([0.6, 2.1, 3.7, 7.8])),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ],
)
def test_populate_fails_type_check(string_data):
    tensor = ov.Tensor(ov.Type.string, ov.Shape([1]))
    assert tensor.element_type == ov.Type.string
    with pytest.raises(RuntimeError) as e:
        tensor.bytes_data = string_data
    assert "Unknown string kind passed to fill the Tensor's data!" in str(e.value)
    with pytest.raises(RuntimeError) as e:
        tensor.str_data = string_data
    assert "Unknown string kind passed to fill the Tensor's data!" in str(e.value)


@pytest.mark.parametrize(
    ("init_type"),
    [
        (ov.Type.string),
        (str),
        (bytes),
        (np.str_),
        (np.bytes_),
    ],
)
@pytest.mark.parametrize(
    ("init_shape", "string_data"),
    [
        (ov.Shape([3]), np.array(["text", "abc", "openvino"]).astype(np.bytes_)),
        (ov.Shape([3]), np.array(["text", "больше текста", "jeszcze więcej słów"])),
        (ov.Shape([3]), [b"xyz", b"abc", b"this is my last"]),
        (ov.Shape([3]), ["text", "abc", "openvino"]),
        (ov.Shape([2]), [[b"text\0with\0null"], [b"openvino\0"]]),
        (ov.Shape([3]), ["text", "больше текста", "jeszcze więcej słów"]),
        (ov.Shape([2, 2]), np.array(["text", "abc", "openvino", "different"]).astype(np.bytes_)),
        (ov.Shape([2, 2]), np.array(["text", "больше текста", "jeszcze więcej słów", "abcdefg"])),
        (ov.Shape([2, 2]), [b"xyz", b"abc", b"this is my last", b"this is my final"]),
        (ov.Shape([2, 2]), [["text", "abc"], ["openvino", "abcdefg"]]),
        (ov.Shape([2, 2]), ["text", "больше текста", "jeszcze więcej słów", "śćżó"]),
    ],
)
@pytest.mark.parametrize(
    ("data_getter"),
    [
        (DataGetter.BYTES),
        (DataGetter.STRINGS),
    ],
)
def test_empty_tensor_populate(init_type, init_shape, string_data, data_getter):
    tensor = ov.Tensor(init_type, init_shape)
    assert tensor.element_type == ov.Type.string
    if data_getter == DataGetter.BYTES:
        tensor.bytes_data = string_data
    elif data_getter == DataGetter.STRINGS:
        tensor.str_data = string_data
    else:
        raise AttributeError("Unknown DataGetter passed!")
    _string_data = np.array(string_data) if isinstance(string_data, list) else string_data
    # Need to flatten the numpy array as Tensor can have different shape.
    # It only checks if strings are filling the data correctly.
    # Encoded:
    check_bytes_based(tensor, _string_data, to_flat=True)
    # Decoded:
    check_string_based(tensor, _string_data, to_flat=True)


def test_invalid_bytes_replaced():
    string_data = np.array(b"\xe2\x80")
    tensor = ov.Tensor(string_data, shared_memory=False)

    # Encoded:
    check_bytes_based(tensor, string_data, to_flat=True)
    # Decoded:
    check_string_based(tensor, np.char.decode(string_data, encoding="utf=8", errors="replace"), to_flat=True)
