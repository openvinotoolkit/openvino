# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov

import pytest


def _check_tensor_string(tensor_data, test_data):
    assert tensor_data.shape == test_data.shape
    assert tensor_data.strides == test_data.strides
    assert np.array_equal(tensor_data, test_data)
    assert not (np.shares_memory(tensor_data, test_data))


def check_bytes_based(tensor, string_data):
    tensor_data = tensor.data
    encoded_data = string_data if string_data.dtype.kind == "S" else np.char.encode(string_data)
    assert tensor_data.dtype.kind == "S"
    _check_tensor_string(tensor_data, encoded_data)


def check_string_based(tensor, string_data):
    tensor_data = tensor.data_str
    decoded_data = string_data if string_data.dtype.kind == "U" else np.char.decode(string_data)
    assert tensor_data.dtype.kind == "U"
    _check_tensor_string(tensor_data, decoded_data)


def test_shared_memory_fails():
    data = np.array(["You", "shall", "not", "pass!"])
    with pytest.raises(RuntimeError) as e:
        _ = ov.Tensor(data, shared_memory=True)
    assert "SHARED MEMORY MODE FOR THIS TENSOR IS NOT APPLICABLE! String types can be only copied." in str(e.value)


@pytest.mark.parametrize(
    ("init_type"),
    [
        (ov.Type.string),
        # (str),
        # (bytes),
        # (np.str_),
        # (np.bytes_),
        # (np.dtype("<U8")),
        # (np.dtype(">U4")),
        # (np.dtype("|S6")),
        # (np.float32)
    ],
)
def test_empty_string_tensor(init_type):
    tensor = ov.Tensor(type=init_type, shape=ov.Shape([2, 2]))
    assert tensor.element_type == ov.Type.string

# "ABCDEF",  # ???
# "pojedyńczy ciąg znaków",  # ???
# (np.str_("ABCDEF")),  # "<U6", depending on platform

@pytest.mark.parametrize(
    ("string_data"),
    [
        (np.array(["text", "abc", "openvino"]).astype("S")),  # "|S"
        (np.array([["xyz"], ["abc"]]).astype(np.bytes_)),  # "|S"
        (np.array(["text", "abc", "openvino"])),  # "<U", depending on platform
        (np.array(["text", "больше текста", "jeszcze więcej słów", "효과가 있었어"])),  # "<U", depending on platform
        (np.array([["text"], ["abc"], ["openvino"]])),  # "<U", depending on platform
        (np.array([["jeszcze więcej słów", "효과가 있었어"]])),  # "<U", depending on platform
    ],
)
def test_init_with_numpy_bytes(string_data):
    tensor = ov.Tensor(string_data, shared_memory=False)
    assert tensor.element_type == ov.Type.string
    # Encoded:
    check_bytes_based(tensor, string_data)
    # Decoded:
    check_string_based(tensor, string_data)


@pytest.mark.parametrize(
    ("init_shape"),
    [
        (ov.Shape()),
        (ov.Shape([])),
        (ov.Shape([1, 1])),
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
def test_empty_tensor_populate_bytes(init_shape, string_data):
    tensor = ov.Tensor(ov.Type.string, init_shape)
    assert tensor.element_type == ov.Type.string
    tensor.copy_from(string_data)
    # Encoded:
    check_bytes_based(tensor, string_data)
    # Decoded:
    check_string_based(tensor, string_data)

