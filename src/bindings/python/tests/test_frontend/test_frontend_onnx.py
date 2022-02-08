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
    const_tensor = onnx.helper.make_tensor("const_tensor",
                                           onnx.TensorProto.FLOAT,
                                           (2, 2),
                                           [0.5, 1, 1.5, 2.0])
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


def create_onnx_model_with_custom_attributes():
    add = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"],
                                attribute_i32=np.int32(10),
                                attribute_i64=np.int64(10),
                                attribute_str="string",
                                attribute_f32=np.float(10),
                                attribute_f64=np.float64(10),
                                attribute_bool=np.bool(True),
                                attribute_type=onnx.TensorProto.INT32,

                                attribute_list_i32=np.array([1, 2, 3], dtype=np.int32),
                                attribute_list_i64=np.array([1, 2, 3], dtype=np.int64),
                                attribute_list_str=np.array(["a", "b", "c"], dtype=np.str),
                                attribute_list_f32=np.array([1, 2, 3], dtype=np.float),
                                attribute_list_f64=np.array([1, 2, 3], dtype=np.float64),
                                attribute_list_bool=[True, False, True],
                                attribute_list_type=np.array([onnx.TensorProto.INT32,
                                                              onnx.TensorProto.FLOAT]),

                                )
    const_tensor = onnx.helper.make_tensor("const_tensor",
                                           onnx.TensorProto.FLOAT,
                                           (2, 2),
                                           [0.5, 1, 1.5, 2.0])
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


# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()
onnx_model_filename = "model.onnx"
onnx_model_with_custom_attributes_filename = "model_custom_attributes.onnx"
onnx_model_with_subgraphs_filename = "model_subgraphs.onnx"
ONNX_FRONTEND_NAME = "onnx"


def setup_module():
    onnx.save_model(create_onnx_model(), onnx_model_filename)
    onnx.save_model(create_onnx_model_with_custom_attributes(),
                    onnx_model_with_custom_attributes_filename)
    onnx.save_model(create_onnx_model_with_subgraphs(), onnx_model_with_subgraphs_filename)


def teardown_module():
    os.remove(onnx_model_filename)
    os.remove(onnx_model_with_custom_attributes_filename)
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


def test_onnx_conversion_extension_check_attributes():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.runtime.opset8 as ops

    # use the model with attributes
    fe = fem.load_by_model(onnx_model_with_custom_attributes_filename)
    assert fe
    assert fe.get_name() == "onnx"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True

        def check_attribute(context, name, expected_type, expected_value):
            assert context.has_attribute(name)
            attribute = context.get_attribute(name)
            assert type(attribute) == expected_type
            assert attribute == expected_value

        check_attribute(node, "attribute_i32", int, 10)
        check_attribute(node, "attribute_i64", int, 10)
        check_attribute(node, "attribute_str", str, "string")
        check_attribute(node, "attribute_f32", float, 10.)
        check_attribute(node, "attribute_f64", float, 10.)
        check_attribute(node, "attribute_bool", int, 1)
        check_attribute(node, "attribute_type", int, 6)

        check_attribute(node, "attribute_list_i32", list, [1, 2, 3])
        check_attribute(node, "attribute_list_i64", list, [1, 2, 3])
        check_attribute(node, "attribute_list_str", list, ["a", "b", "c"])
        check_attribute(node, "attribute_list_f32", list, [1., 2., 3.])
        check_attribute(node, "attribute_list_f64", list, [1., 2., 3.])
        check_attribute(node, "attribute_list_bool", list, [1, 0, 1])
        check_attribute(node, "attribute_list_type", list, [6, 1])

        a = node.get_input(0)
        b = node.get_input(1)
        add = ops.add(a, b)
        return [add.output(0)]

    fe.add_extension(ConversionExtension("Add", custom_converter))
    input_model = fe.load(onnx_model_with_custom_attributes_filename)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_onnx_conversion_extension_attribute_with_default_value():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.runtime.opset8 as ops

    # use the model without attributes
    fe = fem.load_by_model(onnx_model_filename)
    assert fe
    assert fe.get_name() == "onnx"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True

        def check_attribute(context, name, default_value):
            assert not context.has_attribute(name)
            attribute = context.get_attribute(name, default_value)
            assert type(attribute) == type(default_value)
            if isinstance(attribute, np.ndarray):
                assert np.all(attribute == default_value)
            else:
                assert attribute == default_value

        check_attribute(node, "attribute_i32", np.int32(5))
        check_attribute(node, "attribute_i64", np.int64(5))
        check_attribute(node, "attribute_str", "abc")
        check_attribute(node, "attribute_f32", np.float32(5))
        check_attribute(node, "attribute_f64", np.float64(5))
        check_attribute(node, "attribute_bool", np.bool(False))
        check_attribute(node, "attribute_type", onnx.TensorProto.FLOAT)

        check_attribute(node, "attribute_list_i32", np.array([4, 5, 6], dtype=np.int32))
        check_attribute(node, "attribute_list_i64", np.array([4, 5, 6], dtype=np.int64))
        check_attribute(node, "attribute_list_str", np.array(["d", "e", "f"], dtype=np.str))
        check_attribute(node, "attribute_list_f32", np.array([4, 5, 6], dtype=np.float))
        check_attribute(node, "attribute_list_f64", np.array([4, 5, 6], dtype=np.float64))
        check_attribute(node, "attribute_list_bool", np.array([True, False, True], dtype=np.bool))
        check_attribute(node, "attribute_list_type", np.array([onnx.TensorProto.INT32,
                                                               onnx.TensorProto.FLOAT]))

        a = node.get_input(0)
        b = node.get_input(1)
        add = ops.add(a, b)
        return [add.output(0)]

    fe.add_extension(ConversionExtension("Add", custom_converter))
    input_model = fe.load(onnx_model_filename)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_onnx_conversion_extension_cast_attributes():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import ConversionExtension
    from openvino.frontend import NodeContext
    from openvino.runtime import Type
    import openvino.runtime.opset8 as ops

    # use the model without attributes
    fe = fem.load_by_model(onnx_model_with_custom_attributes_filename)
    assert fe
    assert fe.get_name() == "onnx"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True

        def check_attribute(context, name, expected_value, dtype):
            attribute = context.get_attribute(name, dtype=dtype)
            if isinstance(attribute, list):
                assert type(attribute[0]) == dtype
            else:
                assert type(attribute) == dtype
            assert attribute == expected_value

        check_attribute(node, "attribute_i32", 10, float)
        check_attribute(node, "attribute_i64", 10, float)
        check_attribute(node, "attribute_str", "string", np.str)
        check_attribute(node, "attribute_f32", 10, int)
        check_attribute(node, "attribute_f64", 10, int)
        check_attribute(node, "attribute_bool", True, bool)
        check_attribute(node, "attribute_type", Type.i32, Type)

        check_attribute(node, "attribute_list_i32", [1., 2., 3.], float)
        check_attribute(node, "attribute_list_i64", [1., 2., 3.], float)
        check_attribute(node, "attribute_list_str", ["a", "b", "c"], np.str)
        check_attribute(node, "attribute_list_f32", [1, 2, 3], int)
        check_attribute(node, "attribute_list_f64", [1, 2, 3], int)
        check_attribute(node, "attribute_list_bool", [True, False, True], bool)
        check_attribute(node, "attribute_list_type", [Type.i32, Type.f32], Type)

        a = node.get_input(0)
        b = node.get_input(1)
        add = ops.add(a, b)
        return [add.output(0)]

    fe.add_extension(ConversionExtension("Add", custom_converter))
    input_model = fe.load(onnx_model_with_custom_attributes_filename)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_onnx_conversion_extension_common():
    skip_if_onnx_frontend_is_disabled()

    # use common (openvino.frontend) import here
    from openvino.frontend import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.runtime.opset8 as ops

    fe = fem.load_by_model(onnx_model_filename)
    assert fe
    assert fe.get_name() == "onnx"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True
        a = node.get_input(0)
        b = node.get_input(1)
        add = ops.add(a, b)
        return [add.output(0)]

    fe.add_extension(ConversionExtension("Add", custom_converter))
    input_model = fe.load(onnx_model_filename)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_onnx_conversion_extension():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.runtime.opset8 as ops

    fe = fem.load_by_model(onnx_model_filename)
    assert fe
    assert fe.get_name() == "onnx"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True
        a = node.get_input(0)
        b = node.get_input(1)
        add = ops.add(a, b)
        return [add.output(0)]

    fe.add_extension(ConversionExtension("Add", custom_converter))
    input_model = fe.load(onnx_model_filename)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_op_extension_via_onnx_extension():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import OpExtension
    from openvino.runtime import Core

    ie = Core()
    ie.add_extension(OpExtension("FW_OV_OP"))
    ie.add_extension(OpExtension("OV_OP", "FW_OP_1"))
    ie.add_extension(OpExtension("OV_OP", "FW_OP_2", {"ov_attribute_1": "fw_attribute_1",
                                                      "ov_attribute_2": "fw_attribute_2"}))
    ie.add_extension(OpExtension("OV_OP", "FW_OP_3", {"ov_attribute_1": "fw_attribute_1",
                                                      "ov_attribute_2": "fw_attribute_2"},
                                 {"ov_attribute_str": "string",
                                  "ov_attribute_int": 4,
                                  "ov_attribute_bool": True,
                                  "ov_attribute_float": 4.,
                                  "ov_attribute_vec_string": ["str1", "str2", "str3"],
                                  "ov_attribute_vec_int": [1, 2, 3, 4, 5, 6, 7],
                                  "ov_attribute_vec_bool": [True, False, True],
                                  "ov_attribute_vec_float": [1., 2., 3., 4., 5., 6., 7.]}))

    model = ie.read_model(onnx_model_filename)
    assert model


def test_op_extension_via_frontend_extension():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend) import here
    from openvino.frontend import OpExtension
    from openvino.runtime import Core

    ie = Core()
    ie.add_extension(OpExtension("FW_OV_OP"))
    ie.add_extension(OpExtension("OV_OP", "FW_OP_1"))
    ie.add_extension(OpExtension("OV_OP", "FW_OP_2", {"ov_attribute_1": "fw_attribute_1",
                                                      "ov_attribute_2": "fw_attribute_2"}))
    ie.add_extension(OpExtension("OV_OP", "FW_OP_3", {"ov_attribute_1": "fw_attribute_1",
                                                      "ov_attribute_2": "fw_attribute_2"},
                                 {"ov_attribute_str": "string",
                                  "ov_attribute_int": 4,
                                  "ov_attribute_bool": True,
                                  "ov_attribute_float": 4.,
                                  "ov_attribute_vec_string": ["str1", "str2", "str3"],
                                  "ov_attribute_vec_int": [1, 2, 3, 4, 5, 6, 7],
                                  "ov_attribute_vec_bool": [True, False, True],
                                  "ov_attribute_vec_float": [1., 2., 3., 4., 5., 6., 7.]}))

    model = ie.read_model(onnx_model_filename)
    assert model
