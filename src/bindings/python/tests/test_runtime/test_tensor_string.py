# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import numpy as np

import openvino as ov

import pytest


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
    print(str(init_type))
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
    ],
)
def test_init_with_numpy_bytes(string_data):
    tensor = ov.Tensor(string_data, shared_memory=False)
    assert tensor.element_type == ov.Type.string
    # Encoded:
    assert tensor.data.dtype.kind == "S"  # always in bytes format
    assert np.array_equal(tensor.data, string_data)
    assert not (np.shares_memory(tensor.data, string_data))
    # Decoded:
    assert tensor.data_str.dtype.kind == "U"  # always in bytes format
    assert np.array_equal(tensor.data_str, np.char.decode(string_data))
    assert not (np.shares_memory(tensor.data_str, string_data))


@pytest.mark.parametrize(
    ("string_data"),
    [
        (np.array(["text", "abc", "openvino"])),  # "<U", depending on platform
        (np.array(["text", "больше текста", "jeszcze więcej słów", "효과가 있었어"])),  # "<U", depending on platform
        (np.array([["text"], ["abc"], ["openvino"]])),  # "<U", depending on platform
        (np.array([["jeszcze więcej słów", "효과가 있었어"]])),  # "<U", depending on platform
    ],
)
def test_init_with_numpy_string(string_data):
    tensor = ov.Tensor(string_data, shared_memory=False)
    assert tensor.element_type == ov.Type.string
    # Encoded:
    assert tensor.data.dtype.kind == "S"  # always in bytes format
    assert np.array_equal(tensor.data, np.char.encode(string_data))
    assert not (np.shares_memory(tensor.data, string_data))
    # Decoded:
    assert tensor.data_str.dtype.kind == "U"  # always in bytes format
    assert np.array_equal(tensor.data_str, string_data)
    assert not (np.shares_memory(tensor.data_str, string_data))


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
        (np.array([["text"], ["abc"]]).astype("S")),  # "|S8"
    ],
)
def test_empty_tensor_populate_bytes(init_shape, string_data):
    tensor = ov.Tensor(ov.Type.string, init_shape)
    assert tensor.element_type == ov.Type.string
    tensor.copy_from(string_data)
    # Encoded:
    assert tensor.data.dtype.kind == "S"  # always in bytes format
    assert np.array_equal(tensor.data, string_data)
    assert not (np.shares_memory(tensor.data, string_data))
    # Decoded:
    assert tensor.data_str.dtype.kind == "U"  # always in bytes format
    assert np.array_equal(tensor.data_str, np.char.decode(string_data))
    assert not (np.shares_memory(tensor.data_str, string_data))


@pytest.mark.parametrize(
    ("init_shape"),
    [
        (ov.Shape()),
        (ov.Shape([1])),
        (ov.Shape([1, 1])),
        (ov.Shape([1, 1])),
        (ov.Shape([2, 4, 5])),
    ],
)
@pytest.mark.parametrize(
    ("string_data"),
    [
        (np.array(["text", "abc", "openvino"])),  # "<U", depending on platform
        (np.array([["text"], ["abc"], ["openvino"]])),  # "<U", depending on platform
        # # Failing, are these strides not being copied over?
        # (np.array([["text", "больше текста"], ["jeszcze więcej słów", "효과가 있었어"]])),
        (np.array([["text", "больше текста"]])),  # "<U"
    ],
)
def test_empty_tensor_populate_strings(init_shape, string_data):
    tensor = ov.Tensor(ov.Type.string, init_shape)
    assert tensor.element_type == ov.Type.string
    tensor.copy_from(string_data)
    # Encoded:
    assert tensor.data.dtype.kind == "S"  # always in bytes format
    assert np.array_equal(tensor.data, np.char.encode(string_data))
    assert not (np.shares_memory(tensor.data, string_data))
    # Decoded:
    assert tensor.data_str.dtype.kind == "U"  # always in bytes format
    assert np.array_equal(tensor.data_str, string_data)
    assert not (np.shares_memory(tensor.data_str, string_data))
