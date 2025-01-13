# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset14 as ops
from openvino import PartialShape, Type


@pytest.mark.parametrize(
    ("input_shape", "adjoint", "expected_output_shape"),
    [
        ([4, 4], False, PartialShape([4, 4])),
        ([10, 8, 8], True, PartialShape([10, 8, 8])),
        ([-1, -1, -1], True, PartialShape([-1, -1, -1])),
        ([10, -1, -1], True, PartialShape([10, -1, -1])),
    ],
)
@pytest.mark.parametrize("op_name", ["inverse", "inverseOpset14"])
def test_inverse_param_inputs(input_shape, adjoint, expected_output_shape, op_name):
    data = ops.parameter(input_shape, dtype=np.float32)

    op = ops.inverse(data, adjoint=adjoint, name=op_name)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "Inverse"
    assert op.get_friendly_name() == op_name
    assert op.get_output_element_type(0) == Type.f32
    assert op.get_output_partial_shape(0) == expected_output_shape


@pytest.mark.parametrize(
    ("input_array", "adjoint", "expected_output_shape"),
    [
        (np.array([[0.7, 0.3], [0.6, 0.5]]), True, PartialShape([2, 2])),
        (np.array([[0.7, 0.3, 0.6], [1, 2, 3], [0.7, 0.1, 0.4]]), False, PartialShape([3, 3])),
    ],
)
@pytest.mark.parametrize("op_name", ["inverse", "inverseOpset14"])
def test_inverse_const_inputs(input_array, adjoint, expected_output_shape, op_name):
    data = ops.constant(input_array, dtype=np.float64)

    op = ops.inverse(data, adjoint=adjoint, name=op_name)

    assert op.get_output_size() == 1
    assert op.get_type_name() == "Inverse"
    assert op.get_friendly_name() == op_name
    assert op.get_output_element_type(0) == Type.f64
    assert op.get_output_partial_shape(0) == expected_output_shape


@pytest.mark.parametrize(
    ("input_shape", "expected_output_shape"),
    [
        ([4, 4], PartialShape([4, 4])),
    ],
)
def test_inverse_default_attrs(input_shape, expected_output_shape):
    data = ops.parameter(input_shape, dtype=np.float16)

    op = ops.inverse(data)

    assert op.get_output_size() == 1
    assert op.get_type_name() == "Inverse"
    assert op.get_output_element_type(0) == Type.f16
    assert op.get_output_partial_shape(0) == expected_output_shape
