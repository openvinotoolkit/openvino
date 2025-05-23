# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type
from openvino.opset15 import parameter
import openvino.opset16 as ov
import pytest


@pytest.mark.parametrize("values_dtype", [
    Type.f32,
    Type.f16,
    Type.i32
])
@pytest.mark.parametrize("indices_dtype", [
    Type.i32,
    Type.i64
])
@pytest.mark.parametrize(("values_shape", "indices_shape", "dense_shape_value"), [
    ((5,), (5, 2), [10, 10]),              # Basic case
    ((0,), (0, 2), [5, 5]),                # Empty values
])
def test_sparse_fill_empty_rows(values_dtype, indices_dtype, values_shape, indices_shape, dense_shape_value):
    values = parameter(values_shape, name="values", dtype=values_dtype)
    dense_shape = parameter((len(dense_shape_value),), name="dense_shape", dtype=indices_dtype)
    indices = parameter(indices_shape, name="indices", dtype=indices_dtype)
    default_value = parameter((), name="default_value", dtype=values_dtype)

    node = ov.sparse_fill_empty_rows(values, dense_shape, indices, default_value)

    assert node.get_type_name() == "SparseFillEmptyRows"
    assert node.get_output_size() == 3

    assert node.get_output_element_type(0) == indices_dtype
    assert node.get_output_element_type(1) == values_dtype
    assert node.get_output_element_type(2) == Type.boolean

    assert node.get_output_partial_shape(0).rank == 2
    assert node.get_output_partial_shape(0)[1] == 2
    assert node.get_output_partial_shape(1).rank == 1
    assert node.get_output_partial_shape(2).rank == 1
