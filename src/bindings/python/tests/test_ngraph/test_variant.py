from openvino.pyopenvino import Variant


def test_variant_str():
    var = Variant("test_string")
    assert isinstance(var.value, str)
    assert var == "test_string"


def test_variant_int():
    var = Variant(2137)
    assert isinstance(var.value, int)
    assert var == 2137


def test_variant_float():
    var = Variant(21.37)
    assert isinstance(var.value, float)


def test_variant_string_list():
    var = Variant(["test", "string"])
    assert isinstance(var.value, list)
    assert isinstance(var.value[0], str)
    assert var.value[0] == "test"


def test_variant_int_list():
    v = Variant([21, 37])
    assert isinstance(v.value, list)
    assert len(v.value) == 2
    assert isinstance(v.value[0], int)


def test_variant_float_list():
    v = Variant([21.0, 37.0])
    assert isinstance(v.value, list)
    assert len(v.value) == 2
    assert isinstance(v.value[0], float)


def test_variant_tuple():
    v = Variant((2, 1))
    assert isinstance(v.value, tuple)


def test_variant_bool():
    v = Variant(False)
    assert isinstance(v.value, bool)
    assert v.value is False


def test_variant_dict_str():
    v = Variant({"key": "value"})
    assert isinstance(v.value, dict)
    assert v.value["key"] == "value"


def test_variant_dict_str_int():
    v = Variant({"key": 2})
    assert isinstance(v.value, dict)
    assert v.value["key"] == 2


def test_variant_int_dict():
    v = Variant({1: 2})
    assert isinstance(v.value, dict)
    assert v.value[1] == 2


def test_set_new_value():
    v = Variant(int(1))
    assert isinstance(v.value, int)
    v = Variant("test")
    assert isinstance(v.value, str)
    assert v == "test"
