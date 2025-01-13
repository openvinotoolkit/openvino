# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset8 as ov


@pytest.mark.parametrize(
    ("shape_a", "shape_b", "transpose_a", "transpose_b", "output_shape"),
    [
        # matrix, vector
        ([2, 4], [4], False, False, [2]),
        ([4], [4, 2], False, False, [2]),
        # matrix, matrix
        ([2, 4], [4, 2], False, False, [2, 2]),
        # tensor, vector
        ([2, 4, 5], [5], False, False, [2, 4]),
        # # tensor, matrix
        ([2, 4, 5], [5, 4], False, False, [2, 4, 4]),
        # # tensor, tensor
        ([2, 2, 4], [2, 4, 2], False, False, [2, 2, 2]),
    ],
)
def test_matmul(shape_a, shape_b, transpose_a, transpose_b, output_shape):
    np.random.seed(133391)
    left_input = -100.0 + np.random.rand(*shape_a).astype(np.float32) * 200.0
    right_input = -100.0 + np.random.rand(*shape_b).astype(np.float32) * 200.0

    node = ov.matmul(left_input, right_input, transpose_a, transpose_b)
    assert node.get_type_name() == "MatMul"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == output_shape
