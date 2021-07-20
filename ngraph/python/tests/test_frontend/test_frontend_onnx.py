# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import numpy as np
from onnx.helper import make_graph, make_model, make_tensor_value_info
from sys import platform
import pytest

from ngraph.frontend import FrontEndManager
from tests.runtime import get_runtime


def create_onnx_model():
    add = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"])
    const_tensor = onnx.helper.make_tensor("const_tensor", onnx.TensorProto.FLOAT, (2, 2), [0.5, 1, 1.5, 2.0])
    const_node = onnx.helper.make_node("Constant", [], outputs=["const_node"],
                                       value=const_tensor, name="const_node")
    mul = onnx.helper.make_node("Mul", inputs=["z", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (2, 2))]
    graph = make_graph([add, const_node, mul], "graph", input_tensors, output_tensors)
    return make_model(graph, producer_name="ngraph ONNX Importer")


def run_function(function, *inputs, expected):
    runtime = get_runtime()
    computation = runtime.computation(function)
    actual = computation(*inputs)
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        np.testing.assert_allclose(expected[i], actual[i], rtol=1e-3, atol=1e-6)


fem = None
onnx_model_filename = "model.onnx"


def setup_module():
    if not os.environ.get("OV_FRONTEND_PATH"):
        if os.environ.get("LD_LIBRARY_PATH"):
            os.environ["OV_FRONTEND_PATH"] = os.environ["LD_LIBRARY_PATH"]
    if not os.environ.get("OV_FRONTEND_PATH"):
        raise RuntimeError("Please set OV_FRONTEND_PATH env variable to point "
                           "to directory that has libonnx_ngraph_frontend.so")
    global fem
    fem = FrontEndManager()
    onnx.save_model(create_onnx_model(), onnx_model_filename)


def teardown_module():
    os.remove(onnx_model_filename)


def skip_if_onnx_frontend_is_disabled():
    paths = os.environ["OV_FRONTEND_PATH"].split(":")
    found = False
    if platform == "linux":
        libname = "libonnx_ngraph_frontend.so"
    elif platform == "darwin":
        libname = "libonnx_ngraph_frontend.dylib"
    elif platform == "win32":
        libname = "onnx_ngraph_frontend.dll"
    for path in paths:
        if os.path.exists(os.path.join(path, libname)):
            found = True
    if not found:
        pytest.skip()


def test_get_available_front_ends():
    skip_if_onnx_frontend_is_disabled()

    front_ends = fem.get_available_front_ends()
    assert "onnx" in front_ends


def test_convert():
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load(onnx_model_filename)
    assert model

    function = fe.convert(model)
    assert function

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[2, 3], [4, 5]], dtype=np.float32)
    expected = np.array([[1.5, 5], [10.5, 18]], dtype=np.float32)
    run_function(function, a, b, expected=[expected])


def test_decode_and_convert():
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load(onnx_model_filename)
    assert model

    decoded_function = fe.decode(model)
    assert decoded_function
    for op in decoded_function.get_ordered_ops():
        assert op.get_type_name() in ["Parameter", "Constant", "ONNXFrameworkNode",
                                      "ONNXSubgraphFrameworkNode", "Result"]

    function = fe.convert(decoded_function)
    assert function
    for op in function.get_ordered_ops():
        assert op.get_type_name() not in ["ONNXFrameworkNode", "ONNXSubgraphFrameworkNode"]

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[2, 3], [4, 5]], dtype=np.float32)
    expected = np.array([[1.5, 5], [10.5, 18]], dtype=np.float32)
    run_function(function, a, b, expected=[expected])
