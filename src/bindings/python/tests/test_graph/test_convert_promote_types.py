# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset14 as ops
from openvino import Type


@pytest.mark.parametrize(
    ("lhs", "rhs", "promote_unsafe", "pytorch_scalar_promotion", "u64_integer_promotion_target", "expected_output_type"),
    [
        (([], np.float32), ([2], np.float16), False, False, "f32", Type.f32),
        (([], np.float32), ([2], np.float16), True, True, Type.f32, Type.f16),
        (([], np.float32), ([2], np.int8), False, True, "f32", Type.f32),
        (([], np.uint64), ([2], np.int8), True, False, "f64", Type.f64),
    ],
)
def test_convert_promote_types_param_inputs(lhs, rhs, promote_unsafe, pytorch_scalar_promotion, u64_integer_promotion_target, expected_output_type):
    lhs_param = ops.parameter(*lhs)
    rhs_param = ops.parameter(*rhs)

    op = ops.convert_promote_types(lhs_param, rhs_param, promote_unsafe, pytorch_scalar_promotion, u64_integer_promotion_target)
    attrs = op.get_attributes()
    assert attrs.get("promote_unsafe") == promote_unsafe
    assert attrs.get("pytorch_scalar_promotion") == pytorch_scalar_promotion
    if isinstance(u64_integer_promotion_target, Type):
        u64_integer_promotion_target = u64_integer_promotion_target.to_string()
    assert attrs.get("u64_integer_promotion_target") == u64_integer_promotion_target
    assert op.get_output_size() == 2
    assert op.get_type_name() == "ConvertPromoteTypes"
    assert op.get_output_element_type(0) == expected_output_type
    assert op.get_output_element_type(1) == expected_output_type
    assert op.get_output_partial_shape(0) == lhs_param.get_output_partial_shape(0)
    assert op.get_output_partial_shape(1) == rhs_param.get_output_partial_shape(0)


@pytest.mark.parametrize(
    ("lhs", "rhs", "promote_unsafe", "pytorch_scalar_promotion", "u64_integer_promotion_target", "expected_output_type"),
    [
        ((1, np.float32), ([2], np.float16), False, False, "f32", Type.f32),
        ((1, np.float32), ([2], np.float16), True, True, "f32", Type.f16),
        ((1, np.float32), ([2], np.int8), False, True, Type.f32, Type.f32),
        ((1, np.uint64), ([2], np.int8), True, False, Type.f64, Type.f64),
    ],
)
def test_convert_promote_types_const_inputs(lhs, rhs, promote_unsafe, pytorch_scalar_promotion, u64_integer_promotion_target, expected_output_type):
    lhs_param = ops.constant(*lhs)
    rhs_param = ops.constant(*rhs)

    op = ops.convert_promote_types(lhs_param, rhs_param, promote_unsafe, pytorch_scalar_promotion, u64_integer_promotion_target)
    attrs = op.get_attributes()
    assert attrs.get("promote_unsafe") == promote_unsafe
    assert attrs.get("pytorch_scalar_promotion") == pytorch_scalar_promotion
    if isinstance(u64_integer_promotion_target, Type):
        u64_integer_promotion_target = u64_integer_promotion_target.to_string()
    assert attrs.get("u64_integer_promotion_target") == u64_integer_promotion_target
    assert op.get_output_size() == 2
    assert op.get_type_name() == "ConvertPromoteTypes"
    assert op.get_output_element_type(0) == expected_output_type
    assert op.get_output_element_type(1) == expected_output_type
    assert op.get_output_partial_shape(0) == lhs_param.get_output_partial_shape(0)
    assert op.get_output_partial_shape(1) == rhs_param.get_output_partial_shape(0)


@pytest.mark.parametrize(
    ("lhs", "rhs", "expected_output_type"),
    [
        (([4, 4], np.float32), ([2], np.float16), Type.f32),
        (([], np.uint32), ([4, 4], np.int64), Type.i64),
        (([], np.uint8), ([4, 4], np.float16), Type.f16),
    ],
)
def test_convert_promote_types_default_attrs(lhs, rhs, expected_output_type):
    lhs_param = ops.parameter(*lhs)
    rhs_param = ops.parameter(*rhs)
    op = ops.convert_promote_types(lhs_param, rhs_param)
    attrs = op.get_attributes()
    assert not attrs.get("promote_unsafe")
    assert not attrs.get("pytorch_scalar_promotion")
    assert attrs.get("u64_integer_promotion_target") == "f32"
    assert op.get_output_size() == 2
    assert op.get_type_name() == "ConvertPromoteTypes"
    assert op.get_output_element_type(0) == expected_output_type
    assert op.get_output_element_type(1) == expected_output_type
    assert op.get_output_partial_shape(0) == lhs_param.get_output_partial_shape(0)
    assert op.get_output_partial_shape(1) == rhs_param.get_output_partial_shape(0)
