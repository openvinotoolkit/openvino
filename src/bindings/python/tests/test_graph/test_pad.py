# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import openvino.opset13 as ov
from openvino import Type


@pytest.mark.parametrize("pad_mode", [
    "constant", "edge", "reflect", "symmetric",
])
def test_pad_mode(pad_mode):
    pads_begin = np.array([0, 1], dtype=np.int32)
    pads_end = np.array([2, 3], dtype=np.int32)

    input_param = ov.parameter((3, 4), name="input", dtype=np.int32)
    model = ov.pad(input_param, pads_begin, pads_end, pad_mode)

    assert model.get_type_name() == "Pad"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == [5, 8]
    assert model.get_output_element_type(0) == Type.i32


@pytest.mark.parametrize(("pads_begin", "pads_end", "output_shape"), [
    ([-1, -1], [-1, -1], [1, 2]),
    ([2, -1], [-1, 3], [4, 6]),
])
def test_pad_being_and_end(pads_begin, pads_end, output_shape):
    input_param = ov.parameter((3, 4), name="input", dtype=np.int32)
    model = ov.pad(input_param, pads_begin, pads_end, "constant")

    assert model.get_type_name() == "Pad"
    assert model.get_output_size() == 1
    assert list(model.get_output_shape(0)) == output_shape
    assert model.get_output_element_type(0) == Type.i32
