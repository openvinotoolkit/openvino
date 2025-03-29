# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset13 as ops
from openvino import PartialShape, Type


@pytest.mark.parametrize(
    ("data_shape", "scale_shape", "shift_shape", "input_type", "destination_type"),
    [
        ([2, 3, 8, 6], [], None, np.float32, None),
        ([2, 3, -1, 6], [], [], np.float16, "f8e4m3"),
        ([4, 4], [4], [4], np.float16, "f8e5m2"),
    ],
)
def test_fake_convert_param_inputs(data_shape, scale_shape, shift_shape, input_type, destination_type):
    data = ops.parameter(data_shape, dtype=input_type)
    scale = ops.parameter(scale_shape, dtype=input_type)
    input_kwargs = {"data": data, "scale": scale}
    if shift_shape is not None:
        input_kwargs["shift"] = ops.parameter(shift_shape, dtype=input_type)
    if destination_type:
        input_kwargs["destination_type"] = destination_type

    op = ops.fake_convert(**input_kwargs)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "FakeConvert"
    assert op.get_output_partial_shape(0) == PartialShape(data_shape)
    assert op.get_output_element_type(0) == Type(input_type)


@pytest.mark.parametrize(
    ("data_array", "scale_array", "shift_array", "input_type", "destination_type"),
    [
        (np.random.random([2, 3, 8, 6]), np.random.random([]), None, np.float32, None),
        (np.random.random([2, 3, 1, 6]), np.random.random([]), np.random.random([]), np.float16, "f8e4m3"),
        (np.random.random([4, 4]), np.random.random([4]), np.random.random([4]), np.float16, "f8e5m2"),
    ],
)
def test_fake_convert_const_inputs(data_array, scale_array, shift_array, input_type, destination_type):
    data = ops.constant(data_array, dtype=input_type)
    scale = ops.constant(scale_array, dtype=input_type)
    input_kwargs = {"data": data, "scale": scale}
    if shift_array is not None:
        input_kwargs["shift"] = ops.constant(shift_array, dtype=input_type)
    if destination_type:
        input_kwargs["destination_type"] = destination_type

    op = ops.fake_convert(**input_kwargs)
    assert op.get_output_size() == 1
    assert op.get_type_name() == "FakeConvert"
    assert op.get_destination_type() == (destination_type if destination_type else "f8e4m3")
    assert op.get_output_partial_shape(0) == PartialShape(data_array.shape)
    assert op.get_output_element_type(0) == Type(input_type)
