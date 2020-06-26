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

import ngraph as ng
from tests.test_ngraph.util import get_runtime, run_op_node


def test_reverse_sequence():
    input_data = np.array(
        [
            0,
            0,
            3,
            0,
            6,
            0,
            9,
            0,
            1,
            0,
            4,
            0,
            7,
            0,
            10,
            0,
            2,
            0,
            5,
            0,
            8,
            0,
            11,
            0,
            12,
            0,
            15,
            0,
            18,
            0,
            21,
            0,
            13,
            0,
            16,
            0,
            19,
            0,
            22,
            0,
            14,
            0,
            17,
            0,
            20,
            0,
            23,
            0,
        ],
        dtype=np.int32,
    ).reshape([2, 3, 4, 2])
    seq_lenghts = np.array([1, 2, 1, 2], dtype=np.int32)
    batch_axis = 2
    sequence_axis = 1

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    seq_lengths_param = ng.parameter(seq_lenghts.shape, name="sequence lengths", dtype=np.int32)
    model = ng.reverse_sequence(input_param, seq_lengths_param, batch_axis, sequence_axis)

    runtime = get_runtime()
    computation = runtime.computation(model, input_param, seq_lengths_param)
    result = computation(input_data, seq_lenghts)

    expected = np.array(
        [
            0,
            0,
            4,
            0,
            6,
            0,
            10,
            0,
            1,
            0,
            3,
            0,
            7,
            0,
            9,
            0,
            2,
            0,
            5,
            0,
            8,
            0,
            11,
            0,
            12,
            0,
            16,
            0,
            18,
            0,
            22,
            0,
            13,
            0,
            15,
            0,
            19,
            0,
            21,
            0,
            14,
            0,
            17,
            0,
            20,
            0,
            23,
            0,
        ],
    ).reshape([1, 2, 3, 4, 2])
    assert np.allclose(result, expected)


def test_pad_edge():
    input_data = np.arange(1, 13).reshape([3, 4])
    pads_begin = np.array([0, 1], dtype=np.int32)
    pads_end = np.array([2, 3], dtype=np.int32)

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int32)
    model = ng.pad(input_param, pads_begin, pads_end, "edge")

    runtime = get_runtime()
    computation = runtime.computation(model, input_param)
    result = computation(input_data)

    expected = np.array(
        [
            [1, 1, 2, 3, 4, 4, 4, 4],
            [5, 5, 6, 7, 8, 8, 8, 8],
            [9, 9, 10, 11, 12, 12, 12, 12],
            [9, 9, 10, 11, 12, 12, 12, 12],
            [9, 9, 10, 11, 12, 12, 12, 12],
        ]
    )
    assert np.allclose(result, expected)


def test_pad_constant():
    input_data = np.arange(1, 13).reshape([3, 4])
    pads_begin = np.array([0, 1], dtype=np.int32)
    pads_end = np.array([2, 3], dtype=np.int32)

    input_param = ng.parameter(input_data.shape, name="input", dtype=np.int64)
    model = ng.pad(input_param, pads_begin, pads_end, "constant", arg_pad_value=np.array(100, dtype=np.int64))

    runtime = get_runtime()
    computation = runtime.computation(model, input_param)
    result = computation(input_data)

    expected = np.array(
        [
            [100, 1, 2, 3, 4, 100, 100, 100],
            [100, 5, 6, 7, 8, 100, 100, 100],
            [100, 9, 10, 11, 12, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100, 100, 100, 100],
        ]
    )
    assert np.allclose(result, expected)


def test_select():
    cond = [[False, False], [True, False], [True, True]]
    then_node = [[-1, 0], [1, 2], [3, 4]]
    else_node = [[11, 10], [9, 8], [7, 6]]
    excepted = [[11, 10], [1, 8], [3, 4]]

    result = run_op_node([cond, then_node, else_node], ng.select)
    assert np.allclose(result, excepted)
