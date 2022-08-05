# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset8 as ov
from tests.test_ngraph.util import run_op_node


@pytest.mark.parametrize(
    ("shape_a", "shape_b", "transpose_a", "transpose_b"),
    [
        # matrix, vector
        ([2, 4], [4], False, False),
        ([4], [4, 2], False, False),
        # matrix, matrix
        ([2, 4], [4, 2], False, False),
        # tensor, vector
        ([2, 4, 5], [5], False, False),
        # # tensor, matrix
        ([2, 4, 5], [5, 4], False, False),
        # # tensor, tensor
        ([2, 2, 4], [2, 4, 2], False, False),
    ],
)
def test_matmul(shape_a, shape_b, transpose_a, transpose_b):
    np.random.seed(133391)
    left_input = -100.0 + np.random.rand(*shape_a).astype(np.float32) * 200.0
    right_input = -100.0 + np.random.rand(*shape_b).astype(np.float32) * 200.0

    result = run_op_node([left_input, right_input], ov.matmul, transpose_a, transpose_b)

    if transpose_a:
        left_input = np.transpose(left_input)
    if transpose_b:
        right_input = np.transpose(right_input)

    expected = np.matmul(left_input, right_input)
    assert np.allclose(result, expected)
