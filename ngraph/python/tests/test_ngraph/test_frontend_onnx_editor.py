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
    models["override_all_outputs.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for test_override_all_inputs
    input_tensors = [
        make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    graph = make_graph([split], "test_graph", input_tensors, output_tensors)
    models["override_all_inputs.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for test_override_all_inputs 2
    input_tensors = [
        make_tensor_value_info("placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("mul_in_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("mul_in_1", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    expected_split = onnx.helper.make_node("Split", inputs=["mul_in_0", "mul_in_1"], outputs=["out4"])
    expected_mul = onnx.helper.make_node("Mul", inputs=["placeholder_port_0"], outputs=["out1", "out2"], name="split1", axis=0)
    graph = make_graph([expected_split, expected_mul], "test_graph", input_tensors, output_tensors)
    models["override_all_inputs_2.onnx"] = make_model(graph, producer_name="ONNX Importer")

    # Expected for test_override_all_inputs 3
    input_tensors = [
        make_tensor_value_info("in3", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out3", onnx.TensorProto.FLOAT, (2, 2)),
        ]
    graph = make_graph([relu], "test_graph", input_tensors, output_tensors)
    models["override_all_inputs_3.onnx"] = make_model(graph, producer_name="ONNX Importer")

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
    comp_flag, mes = CompareNetworks(IENetwork(res_caps), IENetwork(expected_caps))
    return comp_flag 


def teardown_module():
    pass
    #for name in test_models_names:
        #os.remove(name)


def test_override_all_outputs():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="add_out")
    place2 = model.get_place_by_tensor_name(tensorName="out3")
    model.override_all_outputs(outputs=[place1, place2])
    result_func = fe.convert(model)

    expected_model = fe.load_from_file("override_all_outputs.onnx")
    expected_func = fe.convert(expected_model)

    res = comare_functions(result_func, expected_func)
    assert res

def test_override_all_inputs():
    fe = fem.load_by_framework(framework="onnx")
    assert fe

    model = fe.load_from_file("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    #with pytest.raises(RuntimeError):
    model.override_all_inputs(inputs=[place1])

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
