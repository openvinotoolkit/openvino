# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.runtime.opset8 as ov
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node


def test_onehot():
    runtime = get_runtime()
    param = ov.parameter([3], dtype=np.int32)
    model = ov.one_hot(param, 3, 1, 0, 0)
    computation = runtime.computation(model, param)

    expected = np.eye(3)[np.array([1, 0, 2])]
    input_data = np.array([1, 0, 2], dtype=np.int32)
    result = computation(input_data)
    assert np.allclose(result, expected)


def test_one_hot():
    data = np.array([0, 1, 2], dtype=np.int32)
    depth = 2
    on_value = 5
    off_value = 10
    axis = -1
    excepted = [[5, 10], [10, 5], [10, 10]]

    result = run_op_node([data, depth, on_value, off_value], ov.one_hot, axis)
    assert np.allclose(result, excepted)


def test_range():
    start = 5
    stop = 35
    step = 5

    result = run_op_node([start, stop, step], ov.range)
    assert np.allclose(result, [5, 10, 15, 20, 25, 30])
