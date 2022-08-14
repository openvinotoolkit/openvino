# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import OVAny


def test_any_str():
    string = OVAny("test_string")
    assert isinstance(string.value, str)
    assert string == "test_string"


def test_any_int():
    value = OVAny(2137)
    assert isinstance(value.value, int)
    assert value == 2137


def test_any_float():
    value = OVAny(21.37)
    assert isinstance(value.value, float)


def test_any_string_list():
    str_list = OVAny(["test", "string"])
    assert isinstance(str_list.value, list)
    assert isinstance(str_list[0], str)
    assert str_list[0] == "test"


def test_any_int_list():
    value = OVAny([21, 37])
    assert isinstance(value.value, list)
    assert len(value) == 2
    assert isinstance(value[0], int)


def test_any_float_list():
    value = OVAny([21.0, 37.0])
    assert isinstance(value.value, list)
    assert len(value) == 2
    assert isinstance(value[0], float)


def test_any_bool():
    value = OVAny(False)
    assert isinstance(value.value, bool)
    assert value is not True


def test_any_dict_str():
    value = OVAny({"key": "value"})
    assert isinstance(value.value, dict)
    assert value["key"] == "value"


def test_any_dict_str_int():
    value = OVAny({"key": 2})
    assert isinstance(value.value, dict)
    assert value["key"] == 2


def test_any_int_dict():
    value = OVAny({1: 2})
    assert isinstance(value.value, dict)
    assert value[1] == 2


def test_any_set_new_value():
    value = OVAny(int(1))
    assert isinstance(value.value, int)
    value = OVAny("test")
    assert isinstance(value.value, str)
    assert value == "test"


def test_any_class():
    class TestClass:
        def __init__(self):
            self.text = "test"

    value = OVAny(TestClass())
    assert isinstance(value.value, TestClass)
    assert value.value.text == "test"
