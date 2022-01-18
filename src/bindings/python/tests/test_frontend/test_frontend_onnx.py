# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import numpy as np
from onnx.helper import make_graph, make_model, make_tensor_value_info
import pytest

from openvino.frontend import FrontEndManager
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


def create_onnx_model_with_subgraphs():
    A = onnx.helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, [3])
    B = onnx.helper.make_tensor_value_info("B", onnx.TensorProto.FLOAT, [3])
    add_out = onnx.helper.make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, [3])
    sub_out = onnx.helper.make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, [3])

    add = onnx.helper.make_node("Add", inputs=["A", "B"], outputs=["add_out"])
    sub = onnx.helper.make_node("Sub", inputs=["A", "B"], outputs=["sub_out"])

    then_body = make_graph([add], "then_body", [], [add_out])
    else_body = make_graph([sub], "else_body", [], [sub_out])

    if_node = onnx.helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["res"],
        then_branch=then_body,
        else_branch=else_body
    )
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    res = onnx.helper.make_tensor_value_info("res", onnx.TensorProto.FLOAT, [3])

    graph = make_graph([if_node], "graph", [cond, A, B], [res])
    return make_model(graph, producer_name="ngraph ONNX Importer")


def run_function(function, *inputs, expected):
    runtime = get_runtime()
    computation = runtime.computation(function)
    actual = computation(*inputs)
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        np.testing.assert_allclose(expected[i], actual[i], rtol=1e-3, atol=1e-6)


fem = FrontEndManager()
onnx_model_filename = "model.onnx"
onnx_model_with_subgraphs_filename = "model_subgraphs.onnx"
ONNX_FRONTEND_NAME = "onnx"


def setup_module():
    onnx.save_model(create_onnx_model(), onnx_model_filename)
    onnx.save_model(create_onnx_model_with_subgraphs(), onnx_model_with_subgraphs_filename)


def teardown_module():
    os.remove(onnx_model_filename)
    os.remove(onnx_model_with_subgraphs_filename)


def skip_if_onnx_frontend_is_disabled():
    front_ends = fem.get_available_front_ends()
    if ONNX_FRONTEND_NAME not in front_ends:
        pytest.skip()


def test_convert():
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load(onnx_model_filename)
    assert model

    function = fe.convert(model)
    assert function

    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[2, 3], [4, 5]], dtype=np.float32)
    expected = np.array([[1.5, 5], [10.5, 18]], dtype=np.float32)
    run_function(function, a, b, expected=[expected])


@pytest.mark.parametrize("model_filename, inputs, expected", [
    [onnx_model_filename,
     [np.array([[1, 2], [3, 4]], dtype=np.float32),
      np.array([[2, 3], [4, 5]], dtype=np.float32)],
     np.array([[1.5, 5], [10.5, 18]], dtype=np.float32)],
    [onnx_model_with_subgraphs_filename,
     [np.array(False, dtype=bool),
      np.array([1, 2, 3], dtype=np.float32),
      np.array([2, 3, 5], dtype=np.float32)],
     np.array([-1, -1, -2], dtype=np.float32)],
])
def test_decode_and_convert(model_filename, inputs, expected):
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load(model_filename)
    assert model

    decoded_function = fe.decode(model)
    assert decoded_function

    for op in decoded_function.get_ordered_ops():
        assert op.get_type_name() in ["Parameter", "Constant", "ONNXFrameworkNode",
                                      "ONNXSubgraphFrameworkNode", "Result"]

    fe.convert(decoded_function)
    assert decoded_function
    for op in decoded_function.get_ordered_ops():
        assert op.get_type_name() not in ["ONNXFrameworkNode", "ONNXSubgraphFrameworkNode"]

    run_function(decoded_function, *inputs, expected=[expected])


def test_load_by_model():
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_model(onnx_model_filename)
    assert fe
    assert fe.get_name() == "onnx"
    model = fe.load(onnx_model_filename)
    assert model
    decoded_function = fe.decode(model)
    assert decoded_function

    assert not fem.load_by_model("test.xx")
    assert not fem.load_by_model("onnx.yy")
