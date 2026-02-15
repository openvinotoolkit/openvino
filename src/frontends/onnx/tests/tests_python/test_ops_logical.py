# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx
import pytest

from tests.tests_python.utils import run_node


@pytest.mark.parametrize(
    ("onnx_op", "numpy_func", "data_type"),
    [
        pytest.param("And", np.logical_and, bool),
        pytest.param("Or", np.logical_or, bool),
        pytest.param("Xor", np.logical_xor, bool),
        pytest.param("Equal", np.equal, np.int32),
        pytest.param("Greater", np.greater, np.int32),
        pytest.param("Less", np.less, np.int32),
    ],
)
def test_logical(onnx_op, numpy_func, data_type):
    node = onnx.helper.make_node(onnx_op, inputs=["A", "B"], outputs=["C"], broadcast=1)

    input_a = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]).astype(data_type)
    input_b = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]]).astype(data_type)
    expected_output = numpy_func(input_a, input_b)
    graph_results = run_node(node, [input_a, input_b], opset_version=4)
    assert np.array_equal(graph_results, [expected_output])

    input_a = np.array([[0, 1, -1], [0, 1, -1], [0, 1, -1]]).astype(data_type)
    input_b = np.array(1).astype(data_type)
    expected_output = numpy_func(input_a, input_b)
    graph_results = run_node(node, [input_a, input_b], opset_version=4)
    assert np.array_equal(graph_results, [expected_output])


def test_logical_not():
    input_data = np.array([[False, True, True], [False, True, False], [False, False, True]])
    expected_output = np.logical_not(input_data)

    node = onnx.helper.make_node("Not", inputs=["X"], outputs=["Y"])
    graph_results = run_node(node, [input_data])
    assert np.array_equal(graph_results, [expected_output])
