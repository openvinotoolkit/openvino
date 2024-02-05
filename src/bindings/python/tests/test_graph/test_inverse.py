# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset13 as ops
from openvino import PartialShape, Type


@pytest.mark.parametrize(
    ("input_shape", "adjoint"),
    [
        ([4, 4], False),
        ([10, 8, 8], True),
        ([-1, -1, -1], True),
        ([10, -1, -1], True),
    ],
)
def test_multinomial_param_inputs(input_shape, adjoint):
    input = ops.parameter(input_shape, dtype=np.float32)

    op = ops.inverse(input, adjoint=adjoint)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "Inverse"
    assert op.get_output_element_type(0) == Type.f32
    assert op.get_output_partial_shape(0) == input_shape


@pytest.mark.parametrize(
    ("input_array", "adjoint", ),
    [
        (np.array([[0.7, 0.3], [0.6, 0.5]]), True),
        (np.array([[0.7, 0.3, 0.6], [1, 2, 3], [0.7, 0.1, 0.4]]), False),
    ],
)
def test_multinomial_const_inputs(input_array, adjoint):
    input = ops.constant(input_array, dtype=np.float64)

    op = ops.inverse(input, adjoint=adjoint)

    assert op.get_output_size() == 1
    assert op.get_type_name() == "Inverse"
    assert op.get_output_element_type(0) == Type.f64
    assert op.get_output_partial_shape(0) == input_array.shape


@pytest.mark.parametrize(
    ("input_shape"),
    [
        ([4, 4]),
    ],
)
def test_multinomial_default_attrs(input_shape):
    input = ops.parameter(input_shape, dtype=np.float16)

    op = ops.multinomial(input)

    assert op.get_output_size() == 1
    assert op.get_type_name() == "Inverse"
    assert op.get_output_element_type(0) == Type.f16
    assert op.get_output_partial_shape(0) == input_shape
