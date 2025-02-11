# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

import numpy as np
import onnx
import pytest

from tests.tests_python.utils import run_node


@pytest.mark.parametrize(
    ("onnx_op", "numpy_func"), [("Sum", np.add), ("Min", np.minimum), ("Max", np.maximum)],
)
def test_variadic(onnx_op, numpy_func):
    data = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32),
        np.array([7, 8, 9], dtype=np.int32),
    ]
    node = onnx.helper.make_node(
        onnx_op, inputs=["data_0", "data_1", "data_2"], outputs=["y"],
    )
    expected_output = reduce(numpy_func, data)

    graph_results = run_node(node, data)
    assert np.array_equal(graph_results, [expected_output])


def test_mean():
    data = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5, 6], dtype=np.int32),
        np.array([7, 8, 9], dtype=np.int32),
    ]
    node = onnx.helper.make_node(
        "Mean", inputs=["data_0", "data_1", "data_2"], outputs=["y"],
    )
    expected_output = reduce(np.add, data) / len(data)

    graph_results = run_node(node, data)
    assert np.array_equal(graph_results, [expected_output])
