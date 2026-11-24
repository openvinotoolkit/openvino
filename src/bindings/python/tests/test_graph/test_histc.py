# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino import Type
from openvino.opset15 import parameter
import openvino.opset17 as ov


def test_histc_defaults():
    data = parameter([2, 3, 4], name="data", dtype=np.float32)
    node = ov.histc(data)

    assert node.get_type_name() == "Histc"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [100]
    assert node.get_attributes() == {"bins": 100, "min": 0.0, "max": 0.0, "preserve_input_type": True}


def test_histc_attributes_and_name():
    data = parameter([5], name="data", dtype=np.float32)
    node = ov.histc(data, bins=4, min=0.0, max=8.0, name="hist")

    assert node.get_type_name() == "Histc"
    assert node.get_friendly_name() == "hist"
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == [4]
    assert node.get_attributes() == {"bins": 4, "min": 0.0, "max": 8.0, "preserve_input_type": True}


@pytest.mark.parametrize(
    ("dtype", "preserve_input_type", "expected_type"),
    [
        (np.float32, True, Type.f32),
        (np.int32, True, Type.i32),
        (np.uint8, True, Type.u8),
        (np.uint8, False, Type.i32),
        (np.float32, False, Type.i32),
    ],
)
def test_histc_preserve_input_type(dtype, preserve_input_type, expected_type):
    data = parameter([5], name="data", dtype=dtype)
    node = ov.histc(data, bins=4, preserve_input_type=preserve_input_type)

    assert node.get_output_element_type(0) == expected_type
    assert node.get_attributes()["preserve_input_type"] is preserve_input_type


def test_histc_attributes_are_keyword_only():
    data = parameter([5], name="data", dtype=np.float32)
    with pytest.raises(TypeError):
        ov.histc(data, 4)
