# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type, PartialShape, Dimension
from openvino.opset15 import parameter
import openvino.opset16 as ov
import pytest


@pytest.mark.parametrize("dtype_segment_ids", [
    Type.i32,
    Type.i64
])
@pytest.mark.parametrize("dtype_num_segments", [
    Type.i32,
    Type.i64
])
@pytest.mark.parametrize(("data_shape", "segment_ids_shape"), [
    ((4,), (4,)),
    ((1, 3, 4), (1,)),
    ((3, 1, 2, 5), (3,))
])
def test_segment_max_with_num_segments(dtype_segment_ids, dtype_num_segments, data_shape, segment_ids_shape):
    data = parameter(data_shape, name="data", dtype=Type.f32)
    segment_ids = parameter(segment_ids_shape, name="segment_ids", dtype=dtype_segment_ids)
    num_segments = parameter((), name="num_segments", dtype=dtype_num_segments)
    node = ov.segment_max(data, segment_ids, num_segments, fill_mode="ZERO")

    assert node.get_type_name() == "SegmentMax"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_partial_shape(0) == PartialShape([Dimension.dynamic(), *data_shape[1:]])


@pytest.mark.parametrize("dtype_segment_ids", [
    Type.i32,
    Type.i64
])
@pytest.mark.parametrize(("data_shape", "segment_ids_shape"), [
    ((4,), (4,)),
    ((1, 3, 4), (1,)),
    ((3, 1, 2, 5), (3,))
])
def test_segment_max_without_num_segments(dtype_segment_ids, data_shape, segment_ids_shape):
    data = parameter(data_shape, name="data", dtype=Type.f32)
    segment_ids = parameter(segment_ids_shape, name="segment_ids", dtype=dtype_segment_ids)
    node = ov.segment_max(data, segment_ids, fill_mode="LOWEST")

    assert node.get_type_name() == "SegmentMax"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_partial_shape(0) == PartialShape([Dimension.dynamic(), *data_shape[1:]])
