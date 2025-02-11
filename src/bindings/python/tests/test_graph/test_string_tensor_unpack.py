# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type, PartialShape, Dimension
import openvino.opset15 as ov
import pytest


@pytest.mark.parametrize("input_shape", [
    ((4,)),
    ((1, 3, 4)),
    ((3, 1, 2, 5))
])
def test_string_tensor_unpack(input_shape):
    input_data = ov.parameter(input_shape, name="input_data", dtype=Type.string)
    node = ov.string_tensor_unpack(input_data)

    assert node.get_type_name() == "StringTensorUnpack"
    assert node.get_output_size() == 3
    assert list(node.get_output_shape(0)) == list(input_shape)
    assert list(node.get_output_shape(1)) == list(input_shape)
    assert node.get_output_partial_shape(2) == PartialShape([Dimension.dynamic()])
    assert node.get_output_element_type(0) == Type.i32
    assert node.get_output_element_type(1) == Type.i32
    assert node.get_output_element_type(2) == Type.u8
