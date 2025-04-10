# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type
import openvino.opset15 as ov
import pytest


@pytest.mark.parametrize(("dtype"), [
    Type.i32,
    Type.i64
])
@pytest.mark.parametrize(("indices_shape"), [
    (4,),
    (1, 3, 4),
    (3, 1, 2, 5)
])
def test_string_tensor_pack(dtype, indices_shape):
    begins = ov.parameter(indices_shape, name="input_data", dtype=dtype)
    ends = ov.parameter(indices_shape, name="input_data", dtype=dtype)
    symbols = ov.parameter((100,), name="input_data", dtype=Type.u8)
    node = ov.string_tensor_pack(begins, ends, symbols)

    assert node.get_type_name() == "StringTensorPack"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == list(indices_shape)
    assert node.get_output_element_type(0) == Type.string
