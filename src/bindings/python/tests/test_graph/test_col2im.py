# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type
import openvino.runtime.opset15 as ov
import numpy as np
import pytest


@pytest.mark.parametrize(("input_shape", "output_size", "kernel_size", "expected_shape", "params"), [
    ((3, 4), [2, 2], [1, 1], [3, 2, 2], []),
    ((3, 4), [2, 2], [1, 1], [3, 2, 2], [[1, 1], [1, 1], [0, 0], [0, 0]]),
    ((12, 25), [4, 4], [2, 2], [3, 4, 4], [[2, 2], [1, 1], [3, 3], [3, 3]]),
    ((2, 8, 8), [5, 5], [2, 2], [2, 2, 5, 5], [[2, 2], [1, 1], [0, 2], [0, 2]]),
    ((2, 32, 12), [6, 6], [4, 4], [2, 2, 6, 6], [[2, 2], [2, 2], [4, 3], [4, 3]]),
])
def test_col2im(input_shape, output_size, kernel_size, expected_shape, params):
    input_data = ov.parameter(input_shape, name="input_data", dtype=np.float32)
    output_size = np.array(output_size, np.int32)
    kernel_size = np.array(kernel_size, np.int32)

    node = ov.col2im(input_data, output_size, kernel_size, *params)
    assert node.get_type_name() == "Col2Im"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32
