# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino.exceptions import UserInputError
from openvino.utils.input_validation import (
    _check_value,
    check_valid_attribute,
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64])
def test_is_positive_value_signed_type(dtype):
    assert is_positive_value(dtype(16))
    assert not is_positive_value(dtype(-16))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_is_positive_value_unsigned_type(dtype):
    assert is_positive_value(dtype(16))


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64])
def test_is_non_negative_value_signed_type(dtype):
    assert is_non_negative_value(dtype(16))
    assert is_non_negative_value(dtype(0))
    assert not is_non_negative_value(dtype(-1))
    assert not is_non_negative_value(dtype(-16))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_is_non_negative_value_unsigned_type(dtype):
    assert is_non_negative_value(dtype(16))
    assert is_non_negative_value(dtype(0))


@pytest.mark.parametrize(
    ("value", "val_type"),
    [
        (np.int8(64), np.integer),
        (np.int16(64), np.integer),
        (np.int32(64), np.integer),
        (np.int64(64), np.integer),
        (np.uint8(64), np.unsignedinteger),
        (np.uint16(64), np.unsignedinteger),
        (np.uint32(64), np.unsignedinteger),
        (np.uint64(64), np.unsignedinteger),
        (np.float32(64), np.floating),
        (np.float64(64), np.floating),
    ],
)
def test_check_value(value, val_type):
    def is_even(value):
        return value % 2 == 0

    assert _check_value("TestOp", "test_attr", value, val_type, is_even)


@pytest.mark.parametrize(
    ("value", "val_type"),
    [
        (np.int8(64), np.floating),
        (np.int16(64), np.floating),
        (np.int32(64), np.floating),
        (np.int64(64), np.floating),
        (np.uint8(64), np.floating),
        (np.uint16(64), np.floating),
        (np.uint32(64), np.floating),
        (np.uint64(64), np.floating),
        (np.float32(64), np.integer),
        (np.float64(64), np.integer),
    ],
)
def test_check_value_fail_type(value, val_type):
    try:
        _check_value("TestOp", "test_attr", value, val_type, None)
    except UserInputError:
        pass
    else:
        raise AssertionError("Type validation has unexpectedly passed.")


@pytest.mark.parametrize(
    ("value", "val_type"),
    [
        (np.int8(61), np.integer),
        (np.int16(61), np.integer),
        (np.int32(61), np.integer),
        (np.int64(61), np.integer),
        (np.uint8(61), np.unsignedinteger),
        (np.uint16(61), np.unsignedinteger),
        (np.uint32(61), np.unsignedinteger),
        (np.uint64(61), np.unsignedinteger),
        (np.float32(61), np.floating),
        (np.float64(61), np.floating),
    ],
)
def test_check_value_fail_cond(value, val_type):
    def is_even(value):
        return value % 2 == 0

    try:
        _check_value("TestOp", "test_attr", value, val_type, is_even)
    except UserInputError:
        pass
    else:
        raise AssertionError("Condition validation has unexpectedly passed.")


def test_check_valid_attribute():
    attr_dict = {
        "mode": "bilinear",
        "coefficients": [1, 2, 3, 4, 5],
    }

    assert check_valid_attribute("TestOp", attr_dict, "width", np.unsignedinteger, required=False)
    assert check_valid_attribute("TestOp", attr_dict, "mode", np.str_, required=True)
    assert check_valid_attribute("TestOp", attr_dict, "coefficients", np.integer, required=True)

    try:
        check_valid_attribute("TestOp", attr_dict, "alpha", np.floating, required=True)
    except UserInputError:
        pass
    else:
        raise AssertionError("Validation of missing required attribute has unexpectedly passed.")


def test_check_valid_attributes():
    attr_dict = {
        "mode": "bilinear",
        "coefficients": [1, 2, 3, 4, 5],
    }

    def _is_supported_mode(mode):
        return mode in ["linear", "area", "cubic", "bilinear"]

    requirements = [
        ("width", False, np.unsignedinteger, None),
        ("mode", True, np.str_, _is_supported_mode),
        ("coefficients", True, np.integer, lambda x: x > 0),
        ("alpha", False, np.float64, None),
    ]

    assert check_valid_attributes("TestOp", attr_dict, requirements)

    requirements[3] = ("alpha", True, np.float64, None)
    try:
        check_valid_attributes("TestOp", attr_dict, requirements)
    except UserInputError:
        pass
    else:
        raise AssertionError("Validation of missing required attribute has unexpectedly passed.")
