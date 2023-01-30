# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset8 as ov


@pytest.mark.parametrize(("depth", "on_value", "off_value", "axis", "expected_shape"), [
    (2, 5, 10, -1, [3, 2]),
    (3, 1, 0, 0, [3, 3]),
])
def test_one_hot(depth, on_value, off_value, axis, expected_shape):
    param = ov.parameter([3], dtype=np.int32)
    node = ov.one_hot(param, depth, on_value, off_value, axis)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "OneHot"
    assert list(node.get_output_shape(0)) == expected_shape


def test_range():
    start = 5
    stop = 35
    step = 5

    node = ov.range(start, stop, step)
    assert node.get_output_size() == 1
    assert node.get_type_name() == "Range"
    assert list(node.get_output_shape(0)) == [6]
