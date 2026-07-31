# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type, PartialShape
from openvino.opset15 import parameter
import openvino.opset17 as ov
import pytest


@pytest.mark.parametrize(("a_shape", "b_shape", "out_shape"), [
    ((2, 4, 8), (2, 16, 8), (2, 4, 16)),
    ((3, 5, 8), (3, 8, 8), (3, 5, 8)),
])
def test_grouped_matmul_3d_3d(a_shape, b_shape, out_shape):
    mat_a = parameter(a_shape, name="mat_a", dtype=Type.f32)
    mat_b = parameter(b_shape, name="mat_b", dtype=Type.f32)
    node = ov.grouped_matmul(mat_a, mat_b)

    assert node.get_type_name() == "GroupedMatMul"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_partial_shape(0) == PartialShape(list(out_shape))


@pytest.mark.parametrize("dtype_offsets", [Type.i32, Type.i64])
@pytest.mark.parametrize(("a_shape", "b_shape", "offs_shape"), [
    ((24, 8), (3, 16, 8), (3,)),
    ((16, 8), (2, 8, 8), (2,)),
])
def test_grouped_matmul_2d_3d_with_offsets(dtype_offsets, a_shape, b_shape, offs_shape):
    mat_a = parameter(a_shape, name="mat_a", dtype=Type.f32)
    mat_b = parameter(b_shape, name="mat_b", dtype=Type.f32)
    offsets = parameter(offs_shape, name="offsets", dtype=dtype_offsets)
    node = ov.grouped_matmul(mat_a, mat_b, offsets)

    assert node.get_type_name() == "GroupedMatMul"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_partial_shape(0) == PartialShape([a_shape[0], b_shape[1]])
