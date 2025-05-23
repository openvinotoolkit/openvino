# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import onnx
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from openvino.runtime import Core

from tests.runtime import get_runtime
from tests.tests_python.utils.onnx_helpers import import_onnx_model


def test_import_onnx_function():
    model_path = os.path.join(os.path.dirname(__file__), "models/add_abc.onnx")
    core = Core()
    model = core.read_model(model=model_path)

    dtype = np.float32
    value_a = np.array([1.0], dtype=dtype)
    value_b = np.array([2.0], dtype=dtype)
    value_c = np.array([3.0], dtype=dtype)

    runtime = get_runtime()
    computation = runtime.computation(model)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([6], dtype=dtype))


def test_simple_graph():
    node1 = make_node("Add", ["A", "B"], ["X"], name="add_node1")
    node2 = make_node("Add", ["X", "C"], ["Y"], name="add_node2")
    graph = make_graph(
        [node1, node2],
        "test_graph",
        [
            make_tensor_value_info("A", onnx.TensorProto.FLOAT, [1]),
            make_tensor_value_info("B", onnx.TensorProto.FLOAT, [1]),
            make_tensor_value_info("C", onnx.TensorProto.FLOAT, [1]),
        ],
        [make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])],
    )
    model = make_model(graph, producer_name="OpenVINO ONNX Frontend")

    graph_model_function = import_onnx_model(model)

    runtime = get_runtime()
    computation = runtime.computation(graph_model_function)
    assert np.array_equal(
        computation(
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
            np.array([3], dtype=np.float32),
        )[0],
        np.array([6.0], dtype=np.float32),
    )
    assert np.array_equal(
        computation(
            np.array([4], dtype=np.float32),
            np.array([5], dtype=np.float32),
            np.array([6], dtype=np.float32),
        )[0],
        np.array([15.0], dtype=np.float32),
    )
