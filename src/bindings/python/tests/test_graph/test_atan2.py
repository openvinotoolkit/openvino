# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from openvino import Type, PartialShape
from openvino.opset1 import parameter
import openvino.opset17 as ov


@pytest.mark.parametrize("dtype", [Type.f32, Type.f16, Type.bf16, Type.f64])
def test_atan2_same_shape(dtype):
    input_y = parameter([2, 3], name="y", dtype=dtype)
    input_x = parameter([2, 3], name="x", dtype=dtype)
    node = ov.atan2(input_y, input_x)

    assert node.get_type_name() == "Atan2"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == dtype
    assert node.get_output_partial_shape(0) == PartialShape([2, 3])


def test_atan2_numpy_broadcast():
    input_y = parameter([8, 1, 6, 1], name="y", dtype=Type.f32)
    input_x = parameter([7, 1, 5], name="x", dtype=Type.f32)
    node = ov.atan2(input_y, input_x)

    assert node.get_type_name() == "Atan2"
    assert node.get_output_partial_shape(0) == PartialShape([8, 7, 6, 5])
