# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import numpy as np
from onnx.helper import make_graph, make_model, make_tensor_value_info
from openvino.test_utils import CompareNetworks
import ngraph as ng
from ngraph.frontend import FrontEndManager
from openvino.inference_engine import IENetwork
import pytest

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
def create_test_onnx_models():
    models = {}
    # Input model
    add = onnx.helper.make_node("Add", inputs=["in1", "in2"], outputs=["add_out"])
    split = onnx.helper.make_node("Split", inputs=["add_out"], outputs=["out1", "out2"], name="split1", axis=0)
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
    models["input_model.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for extract_subgraph
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    graph = make_graph([add], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for extract_subgraph 2
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    graph = make_graph([add, relu], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_2.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for extract_subgraph 3
    input_tensors = [
        make_tensor_value_info("out1/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    expected_split = onnx.helper.make_node("Split", inputs=["out1/placeholder_port_0"], outputs=["out1", "out2"], name="split1", axis=0)
    graph = make_graph([expected_split], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_3.onnx"] = make_model(graph, producer_name="ONNX Importer")

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
    expected_split = onnx.helper.make_node("Split", inputs=["out1/placeholder_port_0"], outputs=["out1", "out2"])
    expected_mul = onnx.helper.make_node("Mul", inputs=["out4/placeholder_port_0", "out4/placeholder_port_1"], outputs=["out4"], name="split1", axis=0)
    graph = make_graph([expected_split, expected_mul], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_4.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for test_override_all_outputs
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    graph = make_graph([add, relu], "test_graph", input_tensors, output_tensors)
    models["test_override_all_outputs.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for test_override_all_outputs 2
    input_tensors = [
        make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    graph = make_graph([add, mul], "test_graph", input_tensors, output_tensors)
    models["test_override_all_outputs_2.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for test_override_all_inputs
    input_tensors = [
        make_tensor_value_info("out1/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4/placeholder_port_1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    expected_split = onnx.helper.make_node("Split", inputs=["out1/placeholder_port_0"], outputs=["out1", "out2"])
    expected_mul = onnx.helper.make_node("Mul", inputs=["out4/placeholder_port_0", "out4/placeholder_port_1"], outputs=["out4"], name="split1", axis=0)
    graph = make_graph([expected_split, relu, expected_mul], "test_graph", input_tensors, output_tensors)
    models["test_override_all_inputs.onnx"] = make_model(graph, producer_name="ONNX Importer")

    return models


fem = None
test_models_names = []

def setup_module():
    if not os.environ.get("OV_FRONTEND_PATH"):
        if os.environ.get("LD_LIBRARY_PATH"):
            os.environ["OV_FRONTEND_PATH"] = os.environ["LD_LIBRARY_PATH"]
    if not os.environ.get("OV_FRONTEND_PATH"):
        raise RuntimeError("Please set OV_FRONTEND_PATH env variable to point "
                           "to directory that has libonnx_ngraph_frontend.so")
    global fem
    fem = FrontEndManager()
    models = create_test_onnx_models()
    for name, model in models.items():
        onnx.save_model(model, name)
        test_models_names.append(name)

def comare_functions(result, expected):
    res_caps = ng.Function.to_capsule(result)
    expected_caps = ng.Function.to_capsule(expected)
    comp_flag, _ = CompareNetworks(IENetwork(res_caps), IENetwork(expected_caps))
    return comp_flag 


def teardown_module():
    for name in test_models_names:
        os.remove(name)

def test_extract_subgraph():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="add_out").get_producing_operation(0) #in1
    place2 = model.get_place_by_tensor_name(tensorName="add_out").get_producing_operation(1) #in2
    place3 = model.get_place_by_tensor_name(tensorName="add_out")
    model.extract_subgraph(inputs=[place1, place2], outputs=[place3])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("extract_subgraph.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_extract_subgraph_2():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="add_out")
    place2 = model.get_place_by_tensor_name(tensorName="out3")
    model.extract_subgraph(inputs=[], outputs=[place1, place2])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("extract_subgraph_2.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_extract_subgraph_3():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    place2 = model.get_place_by_tensor_name(tensorName="out1")
    place3 = model.get_place_by_tensor_name(tensorName="out2")
    model.extract_subgraph(inputs=[place1], outputs=[place2, place3])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("extract_subgraph_3.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_extract_subgraph_4():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="out4").get_producing_operation(0)
    place2 = model.get_place_by_tensor_name(tensorName="out4").get_producing_operation(1)
    place3 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    place4 = model.get_place_by_tensor_name(tensorName="out1")
    place5 = model.get_place_by_tensor_name(tensorName="out2")
    place6 = model.get_place_by_tensor_name(tensorName="out4")
    model.extract_subgraph(inputs=[place1, place2, place3], outputs=[place4, place5, place6])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("extract_subgraph_4.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_override_all_outputs():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="add_out")
    place2 = model.get_place_by_tensor_name(tensorName="out3")
    model.override_all_outputs(outputs=[place1, place2])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("test_override_all_outputs.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_override_all_outputs_2():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="out4")
    model.override_all_outputs(outputs=[place1])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("test_override_all_outputs_2.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_override_all_inputs():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = place1 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    place2 = model.get_place_by_tensor_name(tensorName="out4").get_producing_operation(0)
    place3 = model.get_place_by_tensor_name(tensorName="out4").get_producing_operation(1)
    place4 = model.get_place_by_tensor_name(tensorName="in3")
    model.override_all_inputs(inputs=[place1, place2, place3, place4])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("test_override_all_inputs.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_is_input_output():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="in2")
    assert place1.is_input() == True
    assert place1.is_output() == False

    place2 = model.get_place_by_tensor_name(tensorName="out2")
    assert place2.is_input() == False
    assert place2.is_output() == True

    place3 = model.get_place_by_tensor_name(tensorName="add_out")
    assert place3.is_input() == False
    assert place3.is_output() == False

    place4 = place1 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    assert place4.is_input() == False
    assert place4.is_output() == False
