# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ngraph as ng
from tests_compatibility.runtime import get_runtime
from tests_compatibility.test_ngraph.util import run_op_node
from tests_compatibility import xfail_issue_78741


def test_onehot():
    runtime = get_runtime()
    param = ng.parameter([3], dtype=np.int32)
    model = ng.one_hot(param, 3, 1, 0, 0)
    computation = runtime.computation(model, param)

    expected = np.eye(3)[np.array([1, 0, 2])]
    input_data = np.array([1, 0, 2], dtype=np.int32)
    result = computation(input_data)
    assert np.allclose(result, expected)


@xfail_issue_78741
def test_one_hot():
    data = np.array([0, 1, 2], dtype=np.int32)
    depth = 2
    on_value = 5
    off_value = 10
    axis = -1
    excepted = [[5, 10], [10, 5], [10, 10]]

    result = run_op_node([data, depth, on_value, off_value], ng.one_hot, axis)
    assert np.allclose(result, excepted)


@xfail_issue_78741
def test_range():
    start = 5
    stop = 35
    step = 5

    result = run_op_node([start, stop, step], ng.range)
    assert np.allclose(result, [5, 10, 15, 20, 25, 30])
