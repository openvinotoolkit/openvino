# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import OVAny
import pytest


@pytest.mark.parametrize(("value", "data_type"), [
    ("test_string", str),
    (2137, int),
    (21.37, float),
    (False, bool),
    ([1, 2, 3], list),
    ((1, 2, 3), tuple),
    ({"a": "b"}, dict),
])
def test_any(value, data_type):
    ovany = OVAny(value)
    assert isinstance(ovany.value, data_type)
    assert ovany == value
    assert ovany.get() == value


@pytest.mark.parametrize(("values", "data_type"), [
    (["test", "string"], str),
    ([21, 37], int),
    ([21.0, 37.0], float),
])
def test_any_list(values, data_type):
    ovany = OVAny(values)
    assert isinstance(ovany.value, list)
    assert isinstance(ovany[0], data_type)
    assert isinstance(ovany[1], data_type)
    assert len(ovany) == 2
    assert ovany.get() == values


@pytest.mark.parametrize(("value_dict", "value_type", "data_type"), [
    ({"key": "value"}, str, str),
    ({21: 37}, int, int),
    ({21.0: 37.0}, float, float),
])
def test_any_dict(value_dict, value_type, data_type):
    ovany = OVAny(value_dict)
    key = list(value_dict.keys())[0]
    assert isinstance(ovany.value, dict)
    assert ovany[key] == list(value_dict.values())[0]
    assert len(ovany.value) == 1
    assert isinstance(ovany.value[key], value_type)
    assert isinstance(list(value_dict.values())[0], data_type)
    assert ovany.get() == value_dict


def test_any_set_new_value():
    value = OVAny(int(1))
    assert isinstance(value.value, int)
    value = OVAny("test")
    assert isinstance(value.value, str)
    assert value == "test"
    value.set(2.5)
    assert isinstance(value.value, float)
    assert value == 2.5


def test_any_class():
    class TestClass:
        def __init__(self):
            self.text = "test"

    value = OVAny(TestClass())
    assert isinstance(value.value, TestClass)
    assert value.value.text == "test"


@pytest.mark.parametrize(("value", "dtype"), [
    ("some_value", str),
    (31.23456, float),
    (True, bool),
    (42, int),
])
def test_astype(value, dtype):
    ovany = OVAny(value)
    assert ovany.astype(dtype) == value


@pytest.mark.parametrize(("value", "dtype"), [
    (["some_value", "another value"], str),
    ([31.23456, -31.3453], float),
    ([True, False], bool),
    ([42, 21], int),
    ([], None),
])
def test_aslist(value, dtype):
    ovany = OVAny(value)
    assert ovany.aslist(dtype) == value
