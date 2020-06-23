# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import pytest

import ngraph as ng
from tests.util import run_op_node


@pytest.mark.parametrize(
    "shape_a, shape_b, transpose_a, transpose_b",
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

    result = run_op_node([left_input, right_input], ng.matmul, transpose_a, transpose_b)

    if transpose_a:
        left_input = np.transpose(left_input)
    if transpose_b:
        right_input = np.transpose(right_input)

    expected = np.matmul(left_input, right_input)
    assert np.allclose(result, expected)
