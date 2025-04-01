# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from openvino import Type


@pytest.mark.parametrize(("dtype_string", "dtype", "ovtype"), [
    ("float16", np.float16, Type.f16),
    ("float32", np.float32, Type.f32),
    ("float64", np.float64, Type.f64),
    ("int8", np.int8, Type.i8),
    ("int16", np.int16, Type.i16),
    ("int32", np.int32, Type.i32),
    ("int64", np.int64, Type.i64),
    ("uint8", np.uint8, Type.u8),
    ("uint16", np.uint16, Type.u16),
    ("uint32", np.uint32, Type.u32),
    ("uint64", np.uint64, Type.u64),
    ("bool", bool, Type.boolean),
    ("bytes_", np.bytes_, Type.string),
    ("str_", np.str_, Type.string),
    ("bytes", bytes, Type.string),
    ("str", str, Type.string),
    ("|S", np.dtype("|S"), Type.string),
    ("|U", np.dtype("|U"), Type.string),
])
def test_dtype_ovtype_conversion(dtype_string, dtype, ovtype):
    if hasattr(dtype, "kind"):
        assert ovtype.to_dtype() == np.bytes_
    elif issubclass(dtype, (str, np.str_)):
        assert ovtype.to_dtype() == np.bytes_
    else:
        assert ovtype.to_dtype() == dtype
    assert Type(dtype_string) == ovtype
    assert Type(dtype) == ovtype


@pytest.mark.parametrize(("ovtype",
                          "static_flag",
                          "dynamic_flag",
                          "real_flag",
                          "integral_flag",
                          "signed_flag",
                          "quantized_flag",
                          "type_name",
                          "type_size",
                          "type_bitwidth"), [
    (Type.f16, True, False, True, False, True, False, "f16", 2, 16),
    (Type.f32, True, False, True, False, True, False, "f32", 4, 32),
    (Type.f64, True, False, True, False, True, False, "f64", 8, 64),
    (Type.i8, True, False, False, True, True, True, "i8", 1, 8),
    (Type.i16, True, False, False, True, True, False, "i16", 2, 16),
    (Type.i32, True, False, False, True, True, True, "i32", 4, 32),
    (Type.i64, True, False, False, True, True, False, "i64", 8, 64),
    (Type.u8, True, False, False, True, False, True, "u8", 1, 8),
    (Type.u16, True, False, False, True, False, False, "u16", 2, 16),
    (Type.u32, True, False, False, True, False, False, "u32", 4, 32),
    (Type.u64, True, False, False, True, False, False, "u64", 8, 64),
    (Type.boolean, True, False, False, True, True, False, "boolean", 1, 8),
])
def test_basic_ovtypes(ovtype,
                       static_flag,
                       dynamic_flag,
                       real_flag,
                       integral_flag,
                       signed_flag,
                       quantized_flag,
                       type_name,
                       type_size,
                       type_bitwidth):
    assert ovtype.is_static() is static_flag
    assert ovtype.is_dynamic() is dynamic_flag
    assert ovtype.is_real() is real_flag
    assert ovtype.real is real_flag
    assert ovtype.is_integral() is integral_flag
    assert ovtype.integral is integral_flag
    assert ovtype.is_signed() is signed_flag
    assert ovtype.signed is signed_flag
    assert ovtype.is_quantized() is quantized_flag
    assert ovtype.quantized is quantized_flag
    assert ovtype.get_type_name() == type_name
    assert ovtype.type_name == type_name
    assert ovtype.get_size() == type_size
    assert ovtype.size == type_size
    assert ovtype.get_bitwidth() == type_bitwidth
    assert ovtype.bitwidth == type_bitwidth


def test_undefined_ovtype():
    with pytest.warns(DeprecationWarning, match="openvino.Type.undefined is deprecated and will be removed in version 2026.0") as w:
        ov_type = Type.undefined
    assert issubclass(w[0].category, DeprecationWarning)
    assert "openvino.Type.undefined is deprecated and will be removed in version 2026.0" in str(w[0].message)

    assert ov_type.is_static() is False
    assert ov_type.is_dynamic() is True
    assert ov_type.is_real() is False
    assert ov_type.is_integral() is True
    assert ov_type.is_signed() is False
    assert ov_type.is_quantized() is False
    assert ov_type.get_type_name() == "dynamic"
    assert ov_type.size == 0
    assert ov_type.get_size() == 0
    assert ov_type.bitwidth == 0
    assert ov_type.get_bitwidth() == 0


def test_dynamic_ov_type():
    ov_type = Type.dynamic
    assert ov_type.is_static() is False
    assert ov_type.is_dynamic() is True
    assert ov_type.is_real() is False
    assert ov_type.is_integral() is True
    assert ov_type.is_signed() is False
    assert ov_type.is_quantized() is False
    assert ov_type.get_type_name() == "dynamic"
    assert ov_type.size == 0
    assert ov_type.get_size() == 0
    assert ov_type.bitwidth == 0
    assert ov_type.get_bitwidth() == 0


@pytest.mark.parametrize(("ovtype_one", "ovtype_two", "expected"), [
    (Type.dynamic, Type.dynamic, True),
    (Type.f32, Type.dynamic, True),
    (Type.dynamic, Type.f32, True),
    (Type.f32, Type.f32, True),
    (Type.f32, Type.f16, False),
    (Type.i16, Type.f32, False),
])
def test_ovtypes_compatibility(ovtype_one, ovtype_two, expected):
    assert ovtype_one.compatible(ovtype_two) is expected


@pytest.mark.parametrize(("ovtype_one", "ovtype_two", "expected"), [
    (Type.dynamic, Type.dynamic, Type.dynamic),
    (Type.f32, Type.dynamic, Type.f32),
    (Type.dynamic, Type.f32, Type.f32),
    (Type.f32, Type.f32, Type.f32),
    (Type.f32, Type.f16, None),
    (Type.i16, Type.f32, None),
])
def test_ovtypes_merge(ovtype_one, ovtype_two, expected):
    assert ovtype_one.merge(ovtype_two) == expected
