# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_tensor_value_info

from openvino.frontend import FrontEndManager, GeneralFailure
from openvino.runtime import Dimension, PartialShape, Type


# ------Test input model 1------
#       in1        in2        in3
#        |          |          |
#        \          /          |
#         +--------+        +------+
#         |  Add   |        | Relu |
#         +--------+        +------+
#          <add_out>           |
#         /       \\           |
#    +--------+  +-----+      out3
#    | Split  |  | Mul |
#    |(split1)|..|     |
#    +--------+  +-----+
#     /     \       |
#   out1   out2    out4
#
#
# ------Test input model 2------
#       in1        in2
#        |          |
#        \          /
#         +--------+
#         |  Add   |
#         +--------+
#          <add_out>
#             |
#        +--------+
#        | Split  |
#        |(split2)|
#        +--------+
#        /         \
#   <sp_out1>    <sp_out2>
#   +-------+    +-------+
#   |  Abs  |    |  Sin  |
#   | (abs1)|    |       |
#   +------ +    +-------+
#      |             |
#     out1          out2
#
#
# ------Test input model 3------
#    in1         in2
#     |         /   \
#     +--------+     +------+
#     |  Add   |     | Relu |
#     +--------+     +------+
#         |              |
#        out1          out2
#
def create_test_onnx_models():
    models = {}
    # Input model 1
    add = onnx.helper.make_node("Add", inputs=["in1", "in2"], outputs=["add_out"], name="onnx_add_op")
    split = onnx.helper.make_node("Split", inputs=["add_out"],
                                  outputs=["out1", "out2"], name="split1", axis=0)
    relu = onnx.helper.make_node("Relu", inputs=["in3"], outputs=["out3"])
    mul = onnx.helper.make_node("Mul", inputs=["add_out", "add_out"], outputs=["out4"])

    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add, split, relu, mul], "test_graph", input_tensors, output_tensors)
    models["input_model.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                            opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Input model 2
    split_2 = onnx.helper.make_node("Split", inputs=["add_out"],
                                    outputs=["sp_out1", "sp_out2"], name="split2", axis=0)
    absolute = onnx.helper.make_node("Abs", inputs=["sp_out1"], outputs=["out1"], name="abs1")
    sin = onnx.helper.make_node("Sin", inputs=["sp_out2"], outputs=["out2"])

    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
    ]
    graph = make_graph([add, split_2, absolute, sin], "test_graph_2", input_tensors, output_tensors)
    models["input_model_2.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                              opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Input model 3
    add_2 = onnx.helper.make_node("Add", inputs=["in1", "in2"], outputs=["out1"], name="onnx_add_op")
    relu_2 = onnx.helper.make_node("Relu", inputs=["in2"], outputs=["out2"])

    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add_2, relu_2], "test_graph_3", input_tensors, output_tensors)
    models["input_model_3.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                              opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for extract_subgraph
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                 opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for extract_subgraph 2
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add, relu], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_2.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                   opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for extract_subgraph 3
    input_tensors = [
        make_tensor_value_info("out1/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    expected_split = onnx.helper.make_node("Split", inputs=["out1/placeholder_port_0"],
                                           outputs=["out1", "out2"], name="split1", axis=0)
    graph = make_graph([expected_split], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_3.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                   opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for extract_subgraph 4
    input_tensors = [
        make_tensor_value_info("out1/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4/placeholder_port_1", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    expected_split = onnx.helper.make_node("Split", inputs=["out1/placeholder_port_0"],
                                           outputs=["out1", "out2"], name="split1", axis=0)
    expected_mul = onnx.helper.make_node("Mul", inputs=["out4/placeholder_port_0", "out4/placeholder_port_1"],
                                         outputs=["out4"])
    graph = make_graph([expected_split, expected_mul], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_4.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                   opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for extract_subgraph 5
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_5.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                   opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for test_override_all_outputs
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add, relu], "test_graph", input_tensors, output_tensors)
    models["test_override_all_outputs.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                          opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for test_override_all_outputs 2
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add, mul], "test_graph", input_tensors, output_tensors)
    models["test_override_all_outputs_2.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                            opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for test_override_all_outputs 3
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add_2], "test_graph_3", input_tensors, output_tensors)
    models["test_override_all_outputs_3.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                            opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for test_override_all_inputs
    input_tensors = [
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out1/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4/placeholder_port_1", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    expected_split = onnx.helper.make_node("Split", inputs=["out1/placeholder_port_0"],
                                           outputs=["out1", "out2"], name="split1", axis=0)
    expected_mul = onnx.helper.make_node("Mul", inputs=["out4/placeholder_port_0", "out4/placeholder_port_1"],
                                         outputs=["out4"])
    graph = make_graph([expected_split, relu, expected_mul], "test_graph", input_tensors, output_tensors)
    models["test_override_all_inputs.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                         opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for cut_and_add_new_input_edge
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("new_input", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    new_mul = onnx.helper.make_node("Mul", inputs=["new_input", "add_out"], outputs=["out4"])
    graph = make_graph([add, split, relu, new_mul], "test_graph", input_tensors, output_tensors)
    models["cut_and_add_new_input_edge.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                           opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for cut_and_add_new_input_place
    input_tensors = [
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("new_input", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    new_mul = onnx.helper.make_node("Mul", inputs=["new_input", "new_input"], outputs=["out4"])
    new_split = onnx.helper.make_node("Split", inputs=["new_input"],
                                      outputs=["out1", "out2"], name="split1", axis=0)
    graph = make_graph([new_split, relu, new_mul], "test_graph", input_tensors, output_tensors)
    models["cut_and_add_new_input_place.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                            opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for remove_output
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    graph = make_graph([add, relu, split], "test_graph", input_tensors, output_tensors)
    models["remove_output.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                              opset_imports=[onnx.helper.make_opsetid("", 13)])

    # test partial shape
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (8, 16)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (8, 16)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (4, 6)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (4, 16)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (4, 16)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (4, 6)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (8, 16)),
    ]
    graph = make_graph([add, split, relu, mul], "test_graph", input_tensors, output_tensors)
    models["test_partial_shape.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                   opset_imports=[onnx.helper.make_opsetid("", 13)])

    # test place names model
    add = onnx.helper.make_node("Add", inputs=["in1", "in2"], outputs=["add_out"])
    sub = onnx.helper.make_node("Sub", inputs=["in1", "in2"], outputs=["sub_out"])
    split = onnx.helper.make_node("Split", inputs=["add_out"], outputs=["out1", "out2"],
                                  name="split1", axis=0)
    mul = onnx.helper.make_node("Mul", inputs=["one_const", "sub_out"], outputs=["out3"])

    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    value_infos = [
        make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    initializers = [
        onnx.helper.make_tensor("one_const", 1, [1], [1]),
    ]
    graph = make_graph([add, sub, split, mul], "test_graph", input_tensors, output_tensors,
                       value_info=value_infos, initializer=initializers)
    models["test_place_names.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                 opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Input model with integer types
    add = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"])
    const_tensor = onnx.helper.make_tensor("const_tensor",
                                           onnx.TensorProto.INT32,
                                           (2, 2),
                                           [5, 1, 4, 20])
    const_node = onnx.helper.make_node("Constant", [], outputs=["const_node"],
                                       value=const_tensor, name="const_node")
    mul = onnx.helper.make_node("Mul", inputs=["z", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.INT32, (2, 2)),
        make_tensor_value_info("y", onnx.TensorProto.INT32, (2, 2)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (2, 2))]
    graph = make_graph([add, const_node, mul], "graph", input_tensors, output_tensors)
    models["input_model_int32.onnx"] = make_model(graph, producer_name="OpenVINO ONNX Frontend",
                                                  opset_imports=[onnx.helper.make_opsetid("", 13)])

    return models


fem = FrontEndManager()
test_models_names = []
ONNX_FRONTEND_NAME = "onnx"


def setup_module():
    models = create_test_onnx_models()
    for name, model in models.items():
        onnx.save_model(model, name)
        test_models_names.append(name)


def teardown_module():
    for name in test_models_names:
        os.remove(name)


def skip_if_onnx_frontend_is_disabled():
    front_ends = fem.get_available_front_ends()
    if ONNX_FRONTEND_NAME not in front_ends:
        pytest.skip()


# Function to compare OV Models (ops names, types and shapes).
# Note that the functions uses get_ordered_ops, so the topological order of ops should be also preserved.
def compare_models(current, expected):  # noqa: C901 the function is too complex
    result = True
    msg = ""
    if current.get_friendly_name() != expected.get_friendly_name():
        result = False
        msg += "Friendly name of nG Functions not equal. "
        msg += f"Current: {current.get_friendly_name()}, expected: {expected.get_friendly_name()}. "

    current_ops = current.get_ordered_ops()
    expected_ops = expected.get_ordered_ops()

    if len(current_ops) != len(expected_ops):
        result = False
        msg += "Not equal number of ops. "
        msg += f"Current: {len(current_ops)}, expected: {len(expected_ops)}. "

    for i in range(len(current_ops)):
        if (current_ops[i].get_friendly_name() != expected_ops[i].get_friendly_name()
                and current_ops[i].get_type_name() != "Constant"):  # const have different names
            result = False
            msg += "Not equal op name. "
            msg += f"Current: {current_ops[i].get_friendly_name()}, "
            msg += f"expected: {expected_ops[i].get_friendly_name()}. "
        if current_ops[i].get_output_size() != expected_ops[i].get_output_size():
            result = False
            msg += f"Not equal output size of {current_ops[i].get_friendly_name()}. "
        for idx in range(current_ops[i].get_output_size()):
            if current_ops[i].get_output_partial_shape(idx) != expected_ops[i].get_output_partial_shape(idx):
                result = False
                msg += f"Not equal op partial shapes of {current_ops[i].get_friendly_name()}. "
                msg += f"Current: {current_ops[i].get_partial_shape({idx})}, "
                msg += f"expected: {expected_ops[i].get_partial_shape({idx})}. "
            if current_ops[i].get_output_element_type(idx) != expected_ops[i].get_output_element_type(idx):
                result = False
                msg += f"Not equal output element type of {current_ops[i].get_friendly_name()}. "
                msg += f"Current: {current_ops[i].get_output_element_type(idx)}, "
                msg += f"expected: {expected_ops[i].get_output_element_type(idx)}. "

    if not result:
        print(msg)  # noqa: T201

    return result


def test_extract_subgraph():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="add_out").get_input_port(input_port_index=0)  # in1
    place2 = model.get_place_by_tensor_name(tensor_name="add_out").get_input_port(input_port_index=1)  # in2
    place3 = model.get_place_by_tensor_name(tensor_name="add_out")
    model.extract_subgraph(inputs=[place1, place2], outputs=[place3])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("extract_subgraph.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_extract_subgraph_2():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="out3")
    place2 = model.get_place_by_tensor_name(tensor_name="add_out")
    model.extract_subgraph(inputs=[], outputs=[place1, place2])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("extract_subgraph_2.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_extract_subgraph_3():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_input_port(operation_name="split1", input_port_index=0)
    place2 = model.get_place_by_tensor_name(tensor_name="out1")
    place3 = model.get_place_by_tensor_name(tensor_name="out2")
    model.extract_subgraph(inputs=[place1], outputs=[place2, place3])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("extract_subgraph_3.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_extract_subgraph_4():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    out4_tensor = model.get_place_by_tensor_name(tensor_name="out4")
    place1 = model.get_place_by_operation_name_and_input_port(operation_name="split1", input_port_index=0)
    place2 = out4_tensor.get_producing_operation().get_input_port(input_port_index=0)
    place3 = out4_tensor.get_producing_operation().get_input_port(input_port_index=1)
    place4 = model.get_place_by_tensor_name(tensor_name="out1")
    place5 = model.get_place_by_tensor_name(tensor_name="out2")
    place6 = model.get_place_by_tensor_name(tensor_name="out4")
    model.extract_subgraph(inputs=[place1, place2, place3], outputs=[place4, place5, place6])
    result_func = fe.convert(model)

    expected_model = fe.convert(fe.load("extract_subgraph_4.onnx"))

    res = compare_models(result_func, expected_model)
    assert res


def test_extract_subgraph_by_op_place_as_input():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split1")
    out4 = model.get_place_by_tensor_name(tensor_name="out4")
    mul_op = out4.get_producing_operation()
    out1 = model.get_place_by_tensor_name(tensor_name="out1")
    out2 = model.get_place_by_tensor_name(tensor_name="out2")

    model.extract_subgraph(inputs=[split_op, mul_op], outputs=[out1, out2, out4])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("extract_subgraph_4.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_extract_subgraph_by_op_place_as_output():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    in1_tensor = model.get_place_by_tensor_name(tensor_name="in1")
    in2_tensor = model.get_place_by_tensor_name(tensor_name="in2")
    add_out_tensor = model.get_place_by_tensor_name(tensor_name="add_out")
    add_op = add_out_tensor.get_producing_operation()

    model.extract_subgraph(inputs=[in1_tensor, in2_tensor], outputs=[add_op])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("extract_subgraph_5.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_extract_subgraph_by_op_place_as_output_2():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split1")
    out4 = model.get_place_by_tensor_name(tensor_name="out4")
    mul_op = out4.get_producing_operation()

    model.extract_subgraph(inputs=[split_op, mul_op], outputs=[])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("test_override_all_inputs.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_extract_subgraph_by_port_place_as_output():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    add_out_tensor = model.get_place_by_tensor_name(tensor_name="add_out")
    add_op = add_out_tensor.get_producing_operation()
    add_op_out_port = add_op.get_output_port(output_port_index=0)
    in1_tensor = model.get_place_by_tensor_name(tensor_name="in1")
    in2_tensor = model.get_place_by_tensor_name(tensor_name="in2")

    model.extract_subgraph(inputs=[in1_tensor, in2_tensor], outputs=[add_op_out_port])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("extract_subgraph.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_override_all_outputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="out3")
    place2 = model.get_place_by_tensor_name(tensor_name="add_out")
    model.override_all_outputs(outputs=[place1, place2])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("test_override_all_outputs.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_override_all_outputs_2():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="out4")
    model.override_all_outputs(outputs=[place1])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("test_override_all_outputs_2.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_override_all_outputs_3():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model_3.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="out1")
    place2 = model.get_place_by_tensor_name(tensor_name="out1")
    model.override_all_outputs(outputs=[place1, place2])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("test_override_all_outputs_3.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_override_all_outputs_invalid_place():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model_3.onnx")
    assert model

    model2 = fe.load("input_model.onnx")
    assert model2
    invalid_place = model2.get_place_by_tensor_name(tensor_name="out3")

    place1 = model.get_place_by_tensor_name(tensor_name="out1")
    place2 = model.get_place_by_tensor_name(tensor_name="out1")
    model.override_all_outputs(outputs=[place1, place2, invalid_place])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("test_override_all_outputs_3.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_override_all_inputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_input_port(
        operation_name="split1", input_port_index=0)
    out4_tensor = model.get_place_by_tensor_name(tensor_name="out4")
    place2 = out4_tensor.get_producing_operation().get_input_port(input_port_index=0)
    place3 = out4_tensor.get_producing_operation().get_input_port(input_port_index=1)
    place4 = model.get_place_by_tensor_name(tensor_name="in3")
    model.override_all_inputs(inputs=[place1, place2, place3, place4])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("test_override_all_inputs.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_override_all_inputs_invalid_place():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model_3.onnx")
    assert model

    model2 = fe.load("input_model.onnx")
    assert model2

    out3_tensor = model2.get_place_by_tensor_name(tensor_name="out3")
    invalid_place = out3_tensor.get_producing_operation().get_input_port(input_port_index=0)

    out1_tensor = model.get_place_by_tensor_name(tensor_name="out1")
    place1 = out1_tensor.get_producing_operation().get_input_port(input_port_index=0)
    place2 = out1_tensor.get_producing_operation().get_input_port(input_port_index=1)
    model.override_all_inputs(inputs=[place1, place2, invalid_place])
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("input_model_3.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_is_input_output():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="in2")
    assert place1.is_input()
    assert not place1.is_output()

    place2 = model.get_place_by_tensor_name(tensor_name="out2")
    assert not place2.is_input()
    assert place2.is_output()

    place3 = model.get_place_by_tensor_name(tensor_name="add_out")
    assert not place3.is_input()
    assert not place3.is_output()

    place4 = model.get_place_by_operation_name_and_input_port(
        operation_name="split1", input_port_index=0)
    assert not place4.is_input()
    assert not place4.is_output()

    place5 = model.get_place_by_operation_name(operation_name="split1")
    assert not place5.is_input()
    assert not place5.is_output()


def test_set_partial_shape():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="in1")
    model.set_partial_shape(place1, PartialShape([8, 16]))
    place2 = model.get_place_by_tensor_name(tensor_name="in2")
    model.set_partial_shape(place2, PartialShape([8, 16]))
    place3 = model.get_place_by_tensor_name(tensor_name="in3")
    model.set_partial_shape(place3, PartialShape([4, 6]))
    result_model = fe.convert(model)

    expected_model = fe.convert(fe.load("test_partial_shape.onnx"))

    res = compare_models(result_model, expected_model)
    assert res


def test_get_partial_shape():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="in1")
    assert model.get_partial_shape(place1) == PartialShape([2, 2])

    place2 = model.get_place_by_tensor_name(tensor_name="out1")
    assert model.get_partial_shape(place2) == PartialShape([1, 2])

    place3 = model.get_place_by_tensor_name(tensor_name="add_out")
    assert model.get_partial_shape(place3) == PartialShape([2, 2])

    place4 = model.get_place_by_tensor_name(tensor_name="in3")
    model.set_partial_shape(place4, PartialShape([4, 6]))
    assert model.get_partial_shape(place4) == PartialShape([4, 6])
    assert model.get_partial_shape(place2) == PartialShape([1, 2])


def test_get_inputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    in_names = [place.get_names()[0] for place in model.get_inputs()]
    assert in_names == ["in1", "in2", "in3"]


def test_get_outputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    out_names = [place.get_names()[0] for place in model.get_outputs()]
    assert out_names == ["out1", "out2", "out3", "out4"]


def test_is_equal():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="in1")
    assert place1.is_equal(place1)

    place2 = model.get_place_by_tensor_name(tensor_name="out2")
    assert place2.is_equal(place2)

    out4_tensor = model.get_place_by_tensor_name(tensor_name="out4")
    place3 = out4_tensor.get_producing_operation().get_input_port(input_port_index=0)
    place4 = out4_tensor.get_producing_operation().get_input_port(input_port_index=0)
    assert place3.is_equal(place4)

    out1_tensor = model.get_place_by_tensor_name(tensor_name="out1")
    place5 = model.get_place_by_operation_name_and_input_port(operation_name="split1", input_port_index=0)
    place6 = out1_tensor.get_producing_operation().get_input_port(input_port_index=0)
    assert place5.is_equal(place6)

    place7 = model.get_place_by_tensor_name(tensor_name="out4").get_producing_port()
    assert place7.is_equal(place7)

    place8 = model.get_place_by_tensor_name(tensor_name="add_out")
    assert place8.is_equal(place8)

    assert not place1.is_equal(place2)
    assert not place6.is_equal(place7)
    assert not place8.is_equal(place2)

    place9 = model.get_place_by_operation_name(operation_name="split1")
    assert place2.get_producing_operation().is_equal(place9)
    assert not place9.is_equal(place2)


def test_is_equal_data():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="in1")
    assert place1.is_equal_data(place1)

    place2 = model.get_place_by_tensor_name(tensor_name="add_out")
    assert place2.is_equal_data(place2)

    place3 = model.get_place_by_tensor_name(tensor_name="in2")
    assert not place1.is_equal_data(place3)
    assert not place2.is_equal_data(place1)

    place4 = place2.get_producing_port()
    assert place2.is_equal_data(place4)

    out4_tensor = model.get_place_by_tensor_name(tensor_name="out4")
    place5 = out4_tensor.get_producing_operation().get_input_port(input_port_index=0)
    assert place2.is_equal_data(place5)
    assert place4.is_equal_data(place5)

    place6 = out4_tensor.get_producing_operation().get_input_port(input_port_index=1)
    assert place6.is_equal_data(place5)

    place7 = model.get_place_by_operation_name_and_input_port(operation_name="split1", input_port_index=0)
    assert place7.is_equal_data(place7)

    place8 = model.get_place_by_tensor_name(tensor_name="out1")
    place9 = model.get_place_by_tensor_name(tensor_name="out2")
    place10 = place8.get_producing_port()
    assert not place8.is_equal_data(place9)
    assert not place9.is_equal_data(place10)
    assert place8.is_equal_data(place10)


def test_get_place_by_tensor_name():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="out2")
    assert place1

    place2 = model.get_place_by_tensor_name(tensor_name="add_out")
    assert place2

    place3 = model.get_place_by_tensor_name(tensor_name="in1")
    assert place3

    assert not model.get_place_by_tensor_name(tensor_name="0:add_out")


def test_get_place_by_operation_name():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name(operation_name="split1")
    assert place1

    place2 = model.get_place_by_operation_name(operation_name="not_existed")
    assert not place2


def test_get_output_port():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split1")
    place1 = split_op.get_output_port(output_port_index=0)
    place2 = split_op.get_output_port(output_name="out2")

    assert place1.get_target_tensor().get_names()[0] == "out1"
    assert place2.get_target_tensor().get_names()[0] == "out2"

    assert not split_op.get_output_port()
    assert not split_op.get_output_port(output_port_index=3)
    assert not split_op.get_output_port(output_name="not_existed")


def test_get_input_port():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split1")
    place1 = split_op.get_input_port(input_port_index=0)
    assert place1.get_source_tensor().get_names()[0] == "add_out"

    place2 = split_op.get_input_port()
    assert place1.is_equal(place2)

    assert not split_op.get_input_port(input_port_index=1)
    assert not split_op.get_input_port(input_name="not_existed")


def test_add_output_place_is_not_output():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place = model.get_place_by_tensor_name(tensor_name="add_out")
    model.add_output(place)

    out_names = [place.get_names()[0] for place in model.get_outputs()]
    assert out_names == ["out1", "out2", "out3", "out4", "add_out"]


def test_add_output_place_is_output():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    orig_model = fe.convert(model)

    place = model.get_place_by_tensor_name(tensor_name="out1")
    model.add_output(place)

    result_model = fe.convert(model)

    res = compare_models(orig_model, result_model)
    assert res


def test_add_output_place_is_input():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    orig_model = fe.convert(model)

    place = model.get_place_by_tensor_name(tensor_name="in1")
    model.add_output(place)
    result_model = fe.convert(model)

    res = compare_models(orig_model, result_model)
    assert res


def test_get_consuming_ports():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensor_name="add_out")
    add_tensor_consuming_ports = place1.get_consuming_ports()
    assert len(add_tensor_consuming_ports) == 3
    place2 = model.get_place_by_operation_name_and_input_port(operation_name="split1", input_port_index=0)
    assert add_tensor_consuming_ports[0].is_equal(place2)
    out4_tensor = model.get_place_by_tensor_name(tensor_name="out4")
    place3 = out4_tensor.get_producing_operation().get_input_port(input_port_index=0)
    assert add_tensor_consuming_ports[1].is_equal(place3)
    place4 = out4_tensor.get_producing_operation().get_input_port(input_port_index=1)
    assert add_tensor_consuming_ports[2].is_equal(place4)

    add_op_consuming_ports = place1.get_producing_operation().get_consuming_ports()
    assert len(add_op_consuming_ports) == len(add_tensor_consuming_ports)
    for i in range(len(add_op_consuming_ports)):
        assert add_op_consuming_ports[i].is_equal(add_tensor_consuming_ports[i])


def test_get_consuming_ports_2():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model_2.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split2")
    split_op_consuming_ports = split_op.get_consuming_ports()
    assert len(split_op_consuming_ports) == 2
    abs_input_port = model.get_place_by_operation_name(operation_name="abs1").get_input_port(input_port_index=0)
    assert split_op_consuming_ports[0].is_equal(abs_input_port)
    out2_tensor = model.get_place_by_tensor_name(tensor_name="out2")
    sin_input_port = out2_tensor.get_producing_operation().get_input_port(input_port_index=0)
    assert split_op_consuming_ports[1].is_equal(sin_input_port)

    split_out_port_0 = split_op.get_output_port(output_port_index=0)
    split_out_port_0_consuming_ports = split_out_port_0.get_consuming_ports()
    assert len(split_out_port_0_consuming_ports) == 1
    assert split_out_port_0_consuming_ports[0].is_equal(abs_input_port)


def test_get_producing_operation():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model_2.onnx")
    assert model

    split_tensor_out_2 = model.get_place_by_tensor_name(tensor_name="sp_out2")
    split_op = model.get_place_by_operation_name(operation_name="split2")
    assert split_tensor_out_2.get_producing_operation().is_equal(split_op)

    split_op = model.get_place_by_operation_name(operation_name="split2")
    split_out_port_2 = split_op.get_output_port(output_port_index=1)
    assert split_out_port_2.get_producing_operation().is_equal(split_op)


def test_get_producing_operation_2():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model_2.onnx")
    assert model

    abs_op = model.get_place_by_operation_name(operation_name="abs1")
    abs_port_0 = abs_op.get_input_port()
    split_op = model.get_place_by_operation_name(operation_name="split2")
    assert abs_port_0.get_producing_operation().is_equal(split_op)
    assert abs_op.get_producing_operation().is_equal(split_op)

    add_out_tensor = model.get_place_by_tensor_name(tensor_name="add_out")
    add_op = add_out_tensor.get_producing_operation()
    assert not add_op.get_producing_operation()

    split_op_producing_op = split_op.get_producing_operation(input_name="add_out")
    assert split_op_producing_op.is_equal(add_op)

    out2_tensor = model.get_place_by_tensor_name(tensor_name="out2")
    sin_op = out2_tensor.get_producing_operation()
    assert sin_op.get_producing_operation(input_port_index=0).is_equal(split_op)


def test_get_consuming_operations():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model_2.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split2")
    split_op_consuming_ops = split_op.get_consuming_operations()
    abs_op = model.get_place_by_operation_name(operation_name="abs1")
    sin_op = model.get_place_by_tensor_name(tensor_name="out2").get_producing_operation()

    assert len(split_op_consuming_ops) == 2
    assert split_op_consuming_ops[0].is_equal(abs_op)
    assert split_op_consuming_ops[1].is_equal(sin_op)

    split_op_port = split_op.get_input_port(input_port_index=0)
    split_op_port_consuming_ops = split_op_port.get_consuming_operations()

    assert len(split_op_port_consuming_ops) == 1
    assert split_op_port_consuming_ops[0].is_equal(split_op)

    add_out_port = model.get_place_by_tensor_name(tensor_name="add_out").get_producing_port()
    add_out_port_consuming_ops = add_out_port.get_consuming_operations()
    assert len(add_out_port_consuming_ops) == 1
    assert add_out_port_consuming_ops[0].is_equal(split_op)

    sp_out2_tensor = model.get_place_by_tensor_name(tensor_name="sp_out2")
    sp_out2_tensor_consuming_ops = sp_out2_tensor.get_consuming_operations()
    assert len(sp_out2_tensor_consuming_ops) == 1
    assert sp_out2_tensor_consuming_ops[0].is_equal(sin_op)

    out2_tensor = model.get_place_by_tensor_name(tensor_name="out2")
    out2_tensor_consuming_ops = out2_tensor.get_consuming_operations()
    assert len(out2_tensor_consuming_ops) == 0
    out2_port_consuming_ops = out2_tensor.get_producing_port().get_consuming_operations()
    assert len(out2_port_consuming_ops) == 0

    split_out_1_consuming_ops = split_op.get_consuming_operations(output_port_index=1)
    assert len(split_out_1_consuming_ops) == 1
    split_out_sp_out_2_consuming_ops = split_op.get_consuming_operations(output_name="sp_out2")
    assert len(split_out_sp_out_2_consuming_ops) == 1
    assert split_out_1_consuming_ops[0].is_equal(split_out_sp_out_2_consuming_ops[0])
    assert split_out_1_consuming_ops[0].is_equal(sin_op)


def test_get_target_tensor():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model_2.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split2")
    assert not split_op.get_target_tensor()

    split_op_tensor_1 = split_op.get_target_tensor(output_port_index=1)
    sp_out2_tensor = model.get_place_by_tensor_name(tensor_name="sp_out2")
    assert split_op_tensor_1.is_equal(sp_out2_tensor)

    split_tensor_sp_out2 = split_op.get_target_tensor(output_name="sp_out2")
    assert split_tensor_sp_out2.is_equal(split_op_tensor_1)

    abs_op = model.get_place_by_operation_name(operation_name="abs1")
    out1_tensor = model.get_place_by_tensor_name(tensor_name="out1")
    assert abs_op.get_target_tensor().is_equal(out1_tensor)


def test_get_source_tensor():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model_2.onnx")
    assert model

    add_out_tensor = model.get_place_by_tensor_name(tensor_name="add_out")
    add_op = add_out_tensor.get_producing_operation()
    assert not add_op.get_source_tensor()

    add_op_in_tensor_1 = add_op.get_source_tensor(input_port_index=1)
    in2_tensor = model.get_place_by_tensor_name(tensor_name="in2")
    assert add_op_in_tensor_1.is_equal(in2_tensor)

    add_op_in_tensor_in2 = add_op.get_source_tensor(input_name="in2")
    assert add_op_in_tensor_in2.is_equal(in2_tensor)

    split_op = model.get_place_by_operation_name(operation_name="split2")
    assert split_op.get_source_tensor().is_equal(add_out_tensor)


def test_get_producing_port():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe
    model = fe.load("input_model_2.onnx")
    assert model

    split_op = model.get_place_by_operation_name(operation_name="split2")
    split_op_in_port = split_op.get_input_port()
    split_op_in_port_prod_port = split_op_in_port.get_producing_port()

    add_out_tensor = model.get_place_by_tensor_name(tensor_name="add_out")
    add_op = add_out_tensor.get_producing_operation()
    add_op_out_port = add_op.get_output_port()

    assert split_op_in_port_prod_port.is_equal(add_op_out_port)


def test_remove_output():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place = model.get_place_by_tensor_name(tensor_name="out4")
    model.remove_output(place)

    expected_model = fe.convert(fe.load("remove_output.onnx"))
    model_converted = fe.convert(model)

    res = compare_models(model_converted, expected_model)
    assert res


def test_remove_output_when_place_is_input():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place = model.get_place_by_tensor_name(tensor_name="in1")
    model.remove_output(place)

    expected_model = fe.convert(fe.load("input_model.onnx"))
    model_converted = fe.convert(model)

    res = compare_models(model_converted, expected_model)
    assert res


def test_get_place_by_operation_name_and_input_port():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_input_port(operation_name="split1", input_port_index=0)
    sp_out1_tensor = model.get_place_by_tensor_name("out2")
    place2 = sp_out1_tensor.get_producing_operation().get_input_port(input_port_index=0)

    assert place1.is_equal(place2)


def test_get_place_by_operation_name_and_output_port():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model_2.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_output_port(operation_name="split2", output_port_index=0)
    sp_out1_tensor = model.get_place_by_tensor_name("sp_out1")
    place2 = sp_out1_tensor.get_producing_operation().get_output_port(output_port_index=0)

    assert place1.is_equal(place2)


def test_cut_and_add_new_input_place():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place = model.get_place_by_tensor_name(tensor_name="add_out")

    model.cut_and_add_new_input(place, "new_input")

    expected_model = fe.convert(fe.load("cut_and_add_new_input_place.onnx"))
    model_converted = fe.convert(model)

    res = compare_models(model_converted, expected_model)
    assert res


def test_cut_and_add_new_input_edge():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    out4 = model.get_place_by_tensor_name(tensor_name="out4")
    mul_op = out4.get_producing_operation()
    edge_mul0 = mul_op.get_input_port(input_port_index=0)

    model.cut_and_add_new_input(edge_mul0, "new_input")

    expected_model = fe.convert(fe.load("cut_and_add_new_input_edge.onnx"))
    model_converted = fe.convert(model)

    res = compare_models(model_converted, expected_model)
    assert res


def test_set_tensor_value():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    new_values = np.array([[1, 2], [3, 4]], dtype=np.float32)

    place1 = model.get_place_by_tensor_name(tensor_name="in1")
    model.set_tensor_value(place1, new_values)

    model_converted = fe.convert(model)

    iteration = None
    current_ops = model_converted.get_ordered_ops()

    for i in range(len(current_ops)):
        if (current_ops[i].get_friendly_name() == "in1"):
            iteration = i

    assert current_ops[iteration] is not None

    retrieved_data = current_ops[iteration].get_data()
    assert np.allclose(new_values, retrieved_data)


def test_not_supported_methods():
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("test_place_names.onnx")

    with pytest.raises(GeneralFailure) as e:
        model.free_name_for_tensor("add_out")
    assert "not applicable for ONNX model" in str(e.value)


def test_set_name_for_tensor():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("test_place_names.onnx")
    old_name = "add_out"
    new_name = "add_out_new"

    tensor = model.get_place_by_tensor_name(tensor_name=old_name)

    # ignore rename to own name (expect no exception)
    model.set_name_for_tensor(tensor=tensor, new_name=old_name)

    with pytest.raises(RuntimeError) as e:
        model.set_name_for_tensor(tensor=tensor, new_name="")
    assert "name must not be empty" in str(e.value)

    # ONNX model stores tensor info separately for inputs, outputs and between nodes tensors
    with pytest.raises(RuntimeError) as e:
        model.set_name_for_tensor(tensor=tensor, new_name="in1")
    assert "already used by another tensor" in str(e.value)
    with pytest.raises(RuntimeError) as e:
        model.set_name_for_tensor(tensor=tensor, new_name="out1")
    assert "already used by another tensor" in str(e.value)
    with pytest.raises(RuntimeError) as e:
        model.set_name_for_tensor(tensor=tensor, new_name="sub_out")
    assert "already used by another tensor" in str(e.value)

    # actual rename
    model.set_name_for_tensor(tensor=tensor, new_name=new_name)

    new_tensor = model.get_place_by_tensor_name(tensor_name=new_name)
    assert new_tensor
    assert new_tensor.is_equal(tensor)  # previous Place object holds the handle

    old_tensor = model.get_place_by_tensor_name(tensor_name=old_name)
    assert old_tensor is None


def test_set_name_for_operation_with_name():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("test_place_names.onnx")
    old_name = "split1"
    new_name = "split1_new"

    operation = model.get_place_by_operation_name(operation_name=old_name)

    # ignore rename to own name (expect no exception)
    model.set_name_for_operation(operation=operation, new_name=old_name)

    # actual rename
    model.set_name_for_operation(operation=operation, new_name=new_name)

    new_operation = model.get_place_by_operation_name(operation_name=new_name)
    assert new_operation
    assert new_operation.is_equal(operation)  # previous Place object holds the handle

    # Below test passes for models with unique operation names, what is not required by ONNX standard
    # If there were more that one nodes with "split1" name, this test would fail.
    old_operation = model.get_place_by_operation_name(operation_name=old_name)
    assert old_operation is None


def test_set_name_for_operation_without_name():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("test_place_names.onnx")
    output_name = "add_out"
    new_name = "Add_new"

    operation = model.get_place_by_tensor_name(tensor_name=output_name).get_producing_operation()
    # assure the test is performed on node with empty name
    assert not operation.get_names() or len(operation.get_names()) == 0 or not operation.get_names()[0]

    # actual rename
    model.set_name_for_operation(operation=operation, new_name=new_name)

    new_operation = model.get_place_by_tensor_name(tensor_name=output_name).get_producing_operation()
    assert new_operation
    assert new_operation.is_equal(operation)  # previous Place object holds the handle


def test_free_name_for_operation():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("test_place_names.onnx")
    name = "split1"

    # assure non existent names are ignored (expect no exception)
    model.free_name_for_operation("non existent name")

    split1 = model.get_place_by_operation_name(operation_name=name)
    assert split1
    model.free_name_for_operation(name)
    operation = model.get_place_by_operation_name(operation_name=name)
    assert not operation

    new_split1 = model.get_place_by_tensor_name(tensor_name="out1").get_producing_operation()
    assert split1.is_equal(new_split1)


def test_set_name_for_dimension():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("test_place_names.onnx")
    dim_name = "batch_size"

    input1 = model.get_place_by_tensor_name(tensor_name="in1")
    model.set_name_for_dimension(input1, 0, dim_name)
    assert model.get_partial_shape(input1) == PartialShape([-1, 2])

    output1 = model.get_place_by_tensor_name(tensor_name="out1")
    model.set_name_for_dimension(output1, 1, dim_name)
    assert model.get_partial_shape(output1) == PartialShape([1, -1])

    # sub_output rank is 2 so setting dim_name at index 3 extends its rank to 4
    sub_output = model.get_place_by_tensor_name(tensor_name="sub_out")
    model.set_name_for_dimension(sub_output, 3, dim_name)
    assert model.get_partial_shape(sub_output) == PartialShape([2, 2, -1, -1])
    with pytest.raises(RuntimeError) as e:
        model.set_name_for_dimension(input1, 0, "")
    assert "name must not be empty" in str(e.value)

    one_const = model.get_place_by_tensor_name(tensor_name="one_const")
    with pytest.raises(RuntimeError) as e:
        model.set_name_for_dimension(one_const, 0, dim_name)
    assert "ONNX initializer shape dimension cannot be dynamic." in str(e.value)


def test_set_input_partial_shape_using_input_edge():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    add_operator = model.get_place_by_operation_name("onnx_add_op")
    add_input_edge = add_operator.get_input_port(input_port_index=0)
    model.set_partial_shape(add_input_edge, PartialShape([10, 10]))
    add_input_edge = add_operator.get_input_port(input_port_index=1)
    model.set_partial_shape(add_input_edge, PartialShape([1]))

    ov_model = fe.convert(model)
    assert ov_model.input("in1").get_partial_shape() == PartialShape([10, 10])
    assert ov_model.input("in2").get_partial_shape() == PartialShape([1])

    assert ov_model.output("out4").get_partial_shape() == PartialShape([10, 10])


def test_set_partial_shape_with_range():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    input1 = model.get_place_by_tensor_name("in1")
    ranged_shape = PartialShape([Dimension(1, 4), Dimension(2)])
    model.set_partial_shape(input1, ranged_shape)

    ov_model = fe.convert(model)
    assert ov_model.input("in1").get_partial_shape() == ranged_shape


def test_set_partial_shape_with_range_and_cut_it_off():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    input1 = model.get_place_by_tensor_name("in1")
    ranged_shape = PartialShape([Dimension(1, 4), Dimension(2)])
    model.set_partial_shape(input1, ranged_shape)

    add_out = model.get_place_by_tensor_name("add_out")
    model.extract_subgraph(inputs=[add_out], outputs=[])

    ov_model = fe.convert(model)
    for model_input in ov_model.inputs:
        assert model_input.get_partial_shape() != ranged_shape


def test_set_partial_shape_with_range_and_rename_it():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    input1 = model.get_place_by_tensor_name("in1")
    ranged_shape = PartialShape([Dimension(1, 4), Dimension(2)])
    model.set_partial_shape(input1, ranged_shape)
    model.set_name_for_tensor(input1, "new_in1")

    ov_model = fe.convert(model)
    assert ov_model.input("new_in1").get_partial_shape() == ranged_shape


def test_get_partial_shape_using_input_edge():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    add_operator = model.get_place_by_operation_name("onnx_add_op")
    add_input_edge = add_operator.get_input_port(input_port_index=0)

    pshape = model.get_partial_shape(add_input_edge)
    assert pshape == PartialShape([2, 2])


def test_get_partial_shape_using_output_edge():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    add_operator = model.get_place_by_operation_name("onnx_add_op")
    add_output_edge = add_operator.get_output_port(output_port_index=0)

    assert model.get_partial_shape(add_output_edge) == PartialShape([2, 2])

    split_operator = model.get_place_by_tensor_name("out1").get_producing_operation()
    out2_edge = split_operator.get_output_port(output_port_index=1)
    assert model.get_partial_shape(out2_edge) == PartialShape([1, 2])


def test_add_name_for_tensor():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    tensor = model.get_place_by_tensor_name(tensor_name="in2")
    model.add_name_for_tensor(tensor, "extra_name")

    ov_model = fe.convert(model)

    add_input = ov_model.input(1)
    add_input_tensor_names = add_input.get_names()
    assert "in2" in add_input_tensor_names
    assert "extra_name" in add_input_tensor_names


def test_add_two_names_for_tensor():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    tensor = model.get_place_by_tensor_name(tensor_name="in2")
    model.add_name_for_tensor(tensor, "extra_name1")
    model.add_name_for_tensor(tensor, "extra_name2")

    ov_model = fe.convert(model)

    add_input = ov_model.input(1)
    add_input_tensor_names = add_input.get_names()
    assert len(add_input_tensor_names) == 3
    assert "extra_name1" in add_input_tensor_names
    assert "extra_name2" in add_input_tensor_names


def test_add_the_same_name_to_tensor_twice():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    tensor = model.get_place_by_tensor_name(tensor_name="in2")
    model.add_name_for_tensor(tensor, "extra_name")
    model.add_name_for_tensor(tensor, "extra_name")

    ov_model = fe.convert(model)

    add_input = ov_model.input(1)
    add_input_tensor_names = add_input.get_names()
    assert len(add_input_tensor_names) == 2
    assert "extra_name" in add_input_tensor_names


def test_add_name_for_tensor_and_cut_it_off():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model_2.onnx")

    tensor = model.get_place_by_tensor_name(tensor_name="in2")
    model.add_name_for_tensor(tensor, "extra_name")

    split_in = model.get_place_by_operation_name("split2").get_input_port(input_port_index=0)
    model.extract_subgraph(inputs=[split_in], outputs=[])

    ov_model = fe.convert(model)

    model_input = ov_model.input(0)
    input_tensor_names = model_input.get_names()
    assert "extra_name" not in input_tensor_names


def test_add_name_for_tensor_and_override_all_inputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model_2.onnx")

    # test with an InputEdge type of Place
    split_in = model.get_place_by_operation_name("split2").get_input_port(input_port_index=0)
    model.add_name_for_tensor(split_in, "extra_name")
    model.override_all_inputs([split_in])

    ov_model = fe.convert(model)

    model_input = ov_model.input(0)
    input_tensor_names = model_input.get_names()
    assert "extra_name" in input_tensor_names


def test_add_name_for_tensor_and_rename_it():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model_2.onnx")

    tensor = model.get_place_by_tensor_name(tensor_name="in2")
    model.add_name_for_tensor(tensor, "extra_name")
    model.set_name_for_tensor(tensor, "renamed_input")

    ov_model = fe.convert(model)

    model_input = ov_model.input(1)
    input_tensor_names = model_input.get_names()
    assert "renamed_input" in input_tensor_names
    assert "extra_name" in input_tensor_names
    assert "in2" not in input_tensor_names


def test_invalidate_input_place_after_extraction():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    split_op = model.get_place_by_operation_name(operation_name="split1")
    place_to_cut = split_op.get_input_port(input_port_index=0)
    model.extract_subgraph(inputs=[split_op], outputs=[])

    with pytest.raises(GeneralFailure) as e:
        place_to_cut.get_source_tensor()
    assert "The place InputEdge{1, 0} is outdated" in str(e.value)


def test_invalidate_output_place_after_extraction():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    split_op = model.get_place_by_operation_name(operation_name="split1")
    out1 = model.get_place_by_tensor_name(tensor_name="out1")
    out2 = model.get_place_by_tensor_name(tensor_name="out2")
    place_to_cut = model.get_place_by_tensor_name(tensor_name="out3").get_producing_port()
    model.extract_subgraph(inputs=[split_op], outputs=[out1, out2])

    with pytest.raises(GeneralFailure) as e:
        place_to_cut.get_target_tensor()
    assert "The place OutputEdge{2, 0} is outdated" in str(e.value)


def test_invalidate_op_place_after_extraction():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model.onnx")

    add_out_tensor = model.get_place_by_tensor_name(tensor_name="add_out")
    place_to_cut = model.get_place_by_operation_name(operation_name="split1")
    model.override_all_outputs(outputs=[add_out_tensor])

    with pytest.raises(GeneralFailure) as e:
        place_to_cut.get_input_port(input_port_index=0)
    assert "The place split1 is outdated" in str(e.value)


def test_override_cut_inputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model_2.onnx")

    split = model.get_place_by_tensor_name(tensor_name="sp_out1").get_producing_operation()
    place_to_cut = model.get_place_by_tensor_name(tensor_name="add_out").get_consuming_ports()[0]
    model.override_all_inputs(inputs=[split])

    with pytest.raises(GeneralFailure) as e:
        model.override_all_inputs(inputs=[place_to_cut])
    assert "The place InputEdge{1, 0} is outdated" in str(e.value)


def test_override_cut_outputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model_2.onnx")

    add_out = model.get_place_by_tensor_name(tensor_name="add_out")
    place_to_cut = model.get_place_by_tensor_name(tensor_name="sp_out1").get_producing_port()
    model.override_all_outputs(outputs=[add_out])

    with pytest.raises(GeneralFailure) as e:
        model.override_all_outputs(outputs=[place_to_cut])
    assert "The place OutputEdge{1, 0} is outdated" in str(e.value)


def test_get_element_type():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model_2.onnx")

    in1 = model.get_place_by_tensor_name(tensor_name="in1")
    assert model.get_element_type(in1) == Type.f32

    in1_output_edge = in1.get_consuming_ports()[0]
    assert model.get_element_type(in1_output_edge) == Type.f32


def test_get_element_type_int32():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model = fe.load("input_model_int32.onnx")

    x_input = model.get_place_by_tensor_name(tensor_name="x")
    assert model.get_element_type(x_input) == Type.i32

    x_output_edge = x_input.get_consuming_ports()[0]
    assert model.get_element_type(x_output_edge) == Type.i32

    # get_element_type can return the concrete element type only for model inputs
    # for other places, it returns undefined type
    const_node = model.get_place_by_tensor_name(tensor_name="const_node")
    assert model.get_element_type(const_node) == Type.undefined
