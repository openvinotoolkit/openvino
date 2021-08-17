# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_tensor_value_info
from ngraph import PartialShape
from ngraph.frontend import FrontEndManager


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
    models["input_model.onnx"] = make_model(graph, producer_name="ONNX Importer",
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
    models["extract_subgraph.onnx"] = make_model(graph, producer_name="ONNX Importer",
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
    models["extract_subgraph_2.onnx"] = make_model(graph, producer_name="ONNX Importer",
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
    models["extract_subgraph_3.onnx"] = make_model(graph, producer_name="ONNX Importer",
                                                   opset_imports=[onnx.helper.make_opsetid("", 13)])

    # Expected for extract_subgraph 4
    input_tensors = [
        make_tensor_value_info("out4/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out4/placeholder_port_1", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("out1/placeholder_port_0", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out1", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out2", onnx.TensorProto.FLOAT, (1, 2)),
        make_tensor_value_info("out4", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    expected_split = onnx.helper.make_node("Split", inputs=["out1/placeholder_port_0"],
                                           outputs=["out1", "out2"])
    expected_mul = onnx.helper.make_node("Mul", inputs=["out4/placeholder_port_0", "out4/placeholder_port_1"],
                                         outputs=["out4"])
    graph = make_graph([expected_split, expected_mul], "test_graph", input_tensors, output_tensors)
    models["extract_subgraph_4.onnx"] = make_model(graph, producer_name="ONNX Importer",
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
    models["test_override_all_outputs.onnx"] = make_model(graph, producer_name="ONNX Importer",
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
    models["test_override_all_outputs_2.onnx"] = make_model(graph, producer_name="ONNX Importer",
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
                                           outputs=["out1", "out2"])
    expected_mul = onnx.helper.make_node("Mul", inputs=["out4/placeholder_port_0", "out4/placeholder_port_1"],
                                         outputs=["out4"])
    graph = make_graph([expected_split, relu, expected_mul], "test_graph", input_tensors, output_tensors)
    models["test_override_all_inputs.onnx"] = make_model(graph, producer_name="ONNX Importer",
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
    models["test_partial_shape.onnx"] = make_model(graph, producer_name="ONNX Importer",
                                                   opset_imports=[onnx.helper.make_opsetid("", 13)])

    return models


fem = FrontEndManager()
test_models_names = []
ONNX_FRONTEND_NAME = "onnx_experimental"


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


# Function to compare ng Functions (ops names, types and shapes).
# Note that the functions uses get_ordered_ops, so the topological order of ops should be also preserved.
def compare_functions(current, expected):  # noqa: C901 the function is too complex
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
        for j in range(current_ops[i].get_output_size()):
            if current_ops[i].get_output_partial_shape(j) != expected_ops[i].get_output_partial_shape(j):
                result = False
                msg += f"Not equal op partial shapes of {current_ops[i].get_friendly_name()}. "
                msg += f"Current: {current_ops[i].get_partial_shape({j})}, "
                msg += f"expected: {expected_ops[i].get_partial_shape({j})}. "
            if current_ops[i].get_output_element_type(j) != expected_ops[i].get_output_element_type(j):
                result = False
                msg += f"Not equal output element type of {current_ops[i].get_friendly_name()}. "
                msg += f"Current: {current_ops[i].get_output_element_type(j)}, "
                msg += f"expected: {expected_ops[i].get_output_element_type(j)}. "

    if not result:
        print(msg)

    return result


def test_extract_subgraph():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="add_out").get_input_port(inputPortIndex=0)  # in1
    place2 = model.get_place_by_tensor_name(tensorName="add_out").get_input_port(inputPortIndex=1)  # in2
    place3 = model.get_place_by_tensor_name(tensorName="add_out")
    model.extract_subgraph(inputs=[place1, place2], outputs=[place3])
    result_func = fe.convert(model)

    expected_model = fe.load("extract_subgraph.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_extract_subgraph_2():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="add_out")
    place2 = model.get_place_by_tensor_name(tensorName="out3")
    model.extract_subgraph(inputs=[], outputs=[place1, place2])
    result_func = fe.convert(model)

    expected_model = fe.load("extract_subgraph_2.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_extract_subgraph_3():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    place2 = model.get_place_by_tensor_name(tensorName="out1")
    place3 = model.get_place_by_tensor_name(tensorName="out2")
    model.extract_subgraph(inputs=[place1], outputs=[place2, place3])
    result_func = fe.convert(model)

    expected_model = fe.load("extract_subgraph_3.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_extract_subgraph_4():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=0)
    place2 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=1)
    place3 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    place4 = model.get_place_by_tensor_name(tensorName="out1")
    place5 = model.get_place_by_tensor_name(tensorName="out2")
    place6 = model.get_place_by_tensor_name(tensorName="out4")
    model.extract_subgraph(inputs=[place1, place2, place3], outputs=[place4, place5, place6])
    result_func = fe.convert(model)

    expected_model = fe.load("extract_subgraph_4.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_override_all_outputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="add_out")
    place2 = model.get_place_by_tensor_name(tensorName="out3")
    model.override_all_outputs(outputs=[place1, place2])
    result_func = fe.convert(model)

    expected_model = fe.load("test_override_all_outputs.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_override_all_outputs_2():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="out4")
    model.override_all_outputs(outputs=[place1])
    result_func = fe.convert(model)

    expected_model = fe.load("test_override_all_outputs_2.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_override_all_inputs():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_operation_name_and_input_port(
        operationName="split1", inputPortIndex=0)
    place2 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=0)
    place3 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=1)
    place4 = model.get_place_by_tensor_name(tensorName="in3")
    model.override_all_inputs(inputs=[place1, place2, place3, place4])
    result_func = fe.convert(model)

    expected_model = fe.load("test_override_all_inputs.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_override_all_inputs_exceptions():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="in1")
    place2 = model.get_place_by_tensor_name(tensorName="in2")
    place3 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    place4 = model.get_place_by_tensor_name(tensorName="in3")

    with pytest.raises(Exception) as e:
        model.override_all_inputs(inputs=[place1, place2])
    assert "Unexpected number of inputs after override_all_inputs" in str(e)

    with pytest.raises(Exception) as e:
        model.override_all_inputs(inputs=[place3, place4])
    assert "Unexpected number of inputs after override_all_inputs" in str(e)


def test_is_input_output():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="in2")
    assert place1.is_input()
    assert not place1.is_output()

    place2 = model.get_place_by_tensor_name(tensorName="out2")
    assert not place2.is_input()
    assert place2.is_output()

    place3 = model.get_place_by_tensor_name(tensorName="add_out")
    assert not place3.is_input()
    assert not place3.is_output()

    place4 = place1 = model.get_place_by_operation_name_and_input_port(
        operationName="split1", inputPortIndex=0)
    assert not place4.is_input()
    assert not place4.is_output()


def test_set_partial_shape():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="in1")
    model.set_partial_shape(place1, PartialShape([8, 16]))
    place2 = model.get_place_by_tensor_name(tensorName="in2")
    model.set_partial_shape(place2, PartialShape([8, 16]))
    place3 = model.get_place_by_tensor_name(tensorName="in3")
    model.set_partial_shape(place3, PartialShape([4, 6]))
    result_func = fe.convert(model)

    expected_model = fe.load("test_partial_shape.onnx")
    expected_func = fe.convert(expected_model)

    res = compare_functions(result_func, expected_func)
    assert res


def test_get_partial_shape():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="in1")
    assert model.get_partial_shape(place1) == PartialShape([2, 2])

    place2 = model.get_place_by_tensor_name(tensorName="out1")
    assert model.get_partial_shape(place2) == PartialShape([1, 2])

    place3 = model.get_place_by_tensor_name(tensorName="add_out")
    assert model.get_partial_shape(place3) == PartialShape([2, 2])

    place4 = model.get_place_by_tensor_name(tensorName="in3")
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

    place1 = model.get_place_by_tensor_name(tensorName="in1")
    assert place1.is_equal(place1)

    place2 = model.get_place_by_tensor_name(tensorName="out2")
    assert place2.is_equal(place2)

    place3 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=0)
    place4 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=0)
    assert place3.is_equal(place4)

    place5 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    place6 = model.get_place_by_tensor_name(tensorName="out1").get_input_port(inputPortIndex=0)
    assert place5.is_equal(place6)

    place7 = model.get_place_by_tensor_name(tensorName="out4").get_producing_port()
    assert place7.is_equal(place7)

    place8 = model.get_place_by_tensor_name(tensorName="add_out")
    assert place8.is_equal(place8)

    assert not place1.is_equal(place2)
    assert not place6.is_equal(place7)
    assert not place8.is_equal(place2)


def test_is_equal_data():
    skip_if_onnx_frontend_is_disabled()
    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load("input_model.onnx")
    assert model

    place1 = model.get_place_by_tensor_name(tensorName="in1")
    assert place1.is_equal_data(place1)

    place2 = model.get_place_by_tensor_name(tensorName="add_out")
    assert place2.is_equal_data(place2)

    place3 = model.get_place_by_tensor_name(tensorName="in2")
    assert not place1.is_equal_data(place3)
    assert not place2.is_equal_data(place1)

    place4 = place2.get_producing_port()
    assert place2.is_equal_data(place4)

    place5 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=0)
    assert place2.is_equal_data(place5)
    assert place4.is_equal_data(place5)

    place6 = model.get_place_by_tensor_name(tensorName="out4").get_input_port(inputPortIndex=1)
    assert place6.is_equal_data(place5)

    place7 = model.get_place_by_operation_name_and_input_port(operationName="split1", inputPortIndex=0)
    assert place7.is_equal_data(place7)

    place8 = model.get_place_by_tensor_name(tensorName="out1")
    place9 = model.get_place_by_tensor_name(tensorName="out2")
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

    place1 = model.get_place_by_tensor_name(tensorName="out2")
    assert place1

    place2 = model.get_place_by_tensor_name(tensorName="add_out")
    assert place2

    place3 = model.get_place_by_tensor_name(tensorName="in1")
    assert place3

    with pytest.raises(Exception) as e:
        model.get_place_by_tensor_name(tensorName="0:add_out")
    assert "The tensor with name: 0:add_out does not exist in the graph" in str(e)
