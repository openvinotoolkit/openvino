# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import OVAny


def test_any_str():
    var = OVAny("test_string")
    assert isinstance(var.value, str)
    assert var == "test_string"


def test_any_int():
    var = OVAny(2137)
    assert isinstance(var.value, int)
    assert var == 2137


def test_any_float():
    var = OVAny(21.37)
    assert isinstance(var.value, float)


def test_any_string_list():
    var = OVAny(["test", "string"])
    assert isinstance(var.value, list)
    assert isinstance(var[0], str)
    assert var[0] == "test"


def test_any_int_list():
    v = OVAny([21, 37])
    assert isinstance(v.value, list)
    assert len(v) == 2
    assert isinstance(v[0], int)


def test_any_float_list():
    v = OVAny([21.0, 37.0])
    assert isinstance(v.value, list)
    assert len(v) == 2
    assert isinstance(v[0], float)


def test_any_bool():
    v = OVAny(False)
    assert isinstance(v.value, bool)
    assert v is not True


def test_any_dict_str():
    v = OVAny({"key": "value"})
    assert isinstance(v.value, dict)
    assert v["key"] == "value"


def test_any_dict_str_int():
    v = OVAny({"key": 2})
    assert isinstance(v.value, dict)
    assert v["key"] == 2


def test_any_int_dict():
    v = OVAny({1: 2})
    assert isinstance(v.value, dict)
    assert v[1] == 2


def test_any_set_new_value():
    v = OVAny(int(1))
    assert isinstance(v.value, int)
    v = OVAny("test")
    assert isinstance(v.value, str)
    assert v == "test"


def test_any_class():
    class TestClass:
        def __init__(self):
            self.text = "test"

    v = OVAny(TestClass())
    assert isinstance(v.value, TestClass)
    assert v.value.text == "test"
