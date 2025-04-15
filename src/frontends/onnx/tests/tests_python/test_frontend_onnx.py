# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import os
import onnx
import numpy as np
from onnx.helper import make_graph, make_model, make_tensor_value_info
import pytest
from pathlib import Path
from itertools import chain
import tempfile
import shutil

from openvino.frontend import FrontEndManager
from tests.runtime import get_runtime


def create_onnx_model():
    add = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"])
    const_tensor = onnx.helper.make_tensor(
        "const_tensor", onnx.TensorProto.FLOAT, (2, 2), [0.5, 1, 1.5, 2.0]
    )
    const_node = onnx.helper.make_node(
        "Constant", [], outputs=["const_node"], value=const_tensor, name="const_node"
    )
    mul = onnx.helper.make_node("Mul", inputs=["z", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (2, 2))]
    graph = make_graph([add, const_node, mul], "graph", input_tensors, output_tensors)
    return make_model(graph, producer_name="OpenVINO ONNX Frontend")


def create_onnx_model_2():
    relu = onnx.helper.make_node("Relu", inputs=["in"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("in", onnx.TensorProto.FLOAT, (1, 2)),
    ]
    output_tensors = [
        make_tensor_value_info("out", onnx.TensorProto.FLOAT, (1, 2)),
    ]
    graph = make_graph([relu], "test_graph", input_tensors, output_tensors)
    return make_model(graph, producer_name="OpenVINO ONNX Frontend")


def create_onnx_model_with_subgraphs():
    x1 = onnx.helper.make_tensor_value_info("x1", onnx.TensorProto.FLOAT, [3])
    x2 = onnx.helper.make_tensor_value_info("x2", onnx.TensorProto.FLOAT, [3])
    add_out = onnx.helper.make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, [3])
    sub_out = onnx.helper.make_tensor_value_info("sub_out", onnx.TensorProto.FLOAT, [3])

    add = onnx.helper.make_node("Add", inputs=["x1", "x2"], outputs=["add_out"])
    sub = onnx.helper.make_node("Sub", inputs=["x1", "x2"], outputs=["sub_out"])

    then_body = make_graph([add], "then_body", [], [add_out])
    else_body = make_graph([sub], "else_body", [], [sub_out])

    if_node = onnx.helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["res"],
        then_branch=then_body,
        else_branch=else_body,
    )
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    res = onnx.helper.make_tensor_value_info("res", onnx.TensorProto.FLOAT, [3])

    graph = make_graph([if_node], "graph", [cond, x1, x2], [res])
    return make_model(graph, producer_name="OpenVINO ONNX Frontend")


def create_onnx_model_with_custom_attributes():
    add = onnx.helper.make_node(
        "Add",
        inputs=["x", "y"],
        outputs=["z"],
        attribute_i32=np.int32(10),
        attribute_i64=np.int64(10),
        attribute_str="string",
        attribute_f32=float(10),
        attribute_f64=np.float64(10),
        attribute_bool=True,
        attribute_type=onnx.TensorProto.INT32,
        attribute_list_i32=np.array([1, 2, 3], dtype=np.int32),
        attribute_list_i64=np.array([1, 2, 3], dtype=np.int64),
        attribute_list_str=np.array(["a", "b", "c"], dtype=str),
        attribute_list_f32=np.array([1, 2, 3], dtype=float),
        attribute_list_f64=np.array([1, 2, 3], dtype=np.float64),
        attribute_list_bool=[True, False, True],
        attribute_list_type=np.array([onnx.TensorProto.INT32, onnx.TensorProto.FLOAT]),
    )
    const_tensor = onnx.helper.make_tensor(
        "const_tensor", onnx.TensorProto.FLOAT, (2, 2), [0.5, 1, 1.5, 2.0]
    )
    const_node = onnx.helper.make_node(
        "Constant", [], outputs=["const_node"], value=const_tensor, name="const_node"
    )
    mul = onnx.helper.make_node("Mul", inputs=["z", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (2, 2))]
    graph = make_graph([add, const_node, mul], "graph", input_tensors, output_tensors)
    return make_model(graph, producer_name="OpenVINO ONNX Frontend")


def create_onnx_model_for_op_extension():
    # operation with double attribute
    elu = onnx.helper.make_node("Elu", alpha=1.0, inputs=["x"], outputs=["elu"])

    # operation with vector<size_t>, enum, bool attributes
    avg_pool = onnx.helper.make_node(
        "AveragePool",
        kernel_shape=[2, 2],
        auto_pad="SAME_LOWER",
        strides=[2, 2],
        inputs=["elu"],
        outputs=["avg_pool"],
    )

    # operation with no attributes
    floor = onnx.helper.make_node("Floor", inputs=["avg_pool"], outputs=["floor"])

    # operation with int64_t attribute
    concat = onnx.helper.make_node(
        "Concat", axis=0, inputs=["floor", "avg_pool"], outputs=["concat"]
    )

    const_tensor = onnx.helper.make_tensor(
        "const_tensor", onnx.TensorProto.FLOAT, [1], [0.5]
    )

    const_node = onnx.helper.make_node(
        "Constant", [], outputs=["const_node"], value=const_tensor, name="const_node"
    )
    # operation with enum attribute
    mul = onnx.helper.make_node("Mul", inputs=["concat", "const_node"], outputs=["mul"])

    # operation with  ov::element::type (class) attribute
    cast = onnx.helper.make_node(
        "Cast", to=int(onnx.TensorProto.FLOAT), inputs=["mul"], outputs=["out"]
    )
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.FLOAT, (1, 3, 32, 32)),
    ]
    output_tensors = [
        make_tensor_value_info("out", onnx.TensorProto.FLOAT, (3, 3, 32, 32))
    ]
    graph = make_graph(
        [const_node, elu, avg_pool, floor, concat, mul, cast],
        "graph",
        input_tensors,
        output_tensors,
    )
    return make_model(graph, producer_name="OpenVINO ONNX Frontend")


def create_onnx_model_extension_with_custom_domain():
    add = onnx.helper.make_node(
        "CustomAdd", inputs=["x", "y"], outputs=["z"], domain="custom_domain"
    )
    const_tensor = onnx.helper.make_tensor(
        "const_tensor", onnx.TensorProto.FLOAT, (2, 2), [0.5, 1, 1.5, 2.0]
    )
    const_node = onnx.helper.make_node(
        "Constant", [], outputs=["const_node"], value=const_tensor, name="const_node"
    )
    mul = onnx.helper.make_node("Mul", inputs=["z", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (2, 2))]
    graph = make_graph([add, const_node, mul], "graph", input_tensors, output_tensors)
    return make_model(graph, producer_name="OpenVINO ONNX Frontend")


def run_model(model, *inputs, expected):
    runtime = get_runtime()
    computation = runtime.computation(model)
    actual = computation(*inputs)
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        np.testing.assert_allclose(expected[i], actual[i], rtol=1e-3, atol=1e-6)


# FrontEndManager shall be initialized and destroyed after all tests finished
# This is because destroy of FrontEndManager will unload all plugins, no objects shall exist after this
fem = FrontEndManager()
model_stream = io.BytesIO()
onnx_model_filename = "model.onnx"
onnx_model_2_filename = "model2.onnx"
onnx_model_with_custom_attributes_filename = "model_custom_attributes.onnx"
onnx_model_with_subgraphs_filename = "model_subgraphs.onnx"
onnx_model_for_op_extension_test = "model_op_extension.onnx"
onnx_model_extension_with_custom_domain = "model_extension_custom_domain.onnx"
ONNX_FRONTEND_NAME = "onnx"


def setup_module():
    onnx.save_model(create_onnx_model(), onnx_model_filename)
    onnx.save_model(create_onnx_model(), model_stream)
    onnx.save_model(create_onnx_model_2(), onnx_model_2_filename)
    onnx.save_model(
        create_onnx_model_with_custom_attributes(),
        onnx_model_with_custom_attributes_filename,
    )
    onnx.save_model(
        create_onnx_model_with_subgraphs(), onnx_model_with_subgraphs_filename
    )
    onnx.save_model(
        create_onnx_model_for_op_extension(), onnx_model_for_op_extension_test
    )
    onnx.save_model(
        create_onnx_model_extension_with_custom_domain(),
        onnx_model_extension_with_custom_domain,
    )


def teardown_module():
    os.remove(onnx_model_filename)
    os.remove(onnx_model_2_filename)
    os.remove(onnx_model_with_custom_attributes_filename)
    os.remove(onnx_model_with_subgraphs_filename)
    os.remove(onnx_model_for_op_extension_test)
    os.remove(onnx_model_extension_with_custom_domain)


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

    converted_model = fe.convert(model)
    assert converted_model

    input_1 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    input_2 = np.array([[2, 3], [4, 5]], dtype=np.float32)
    expected = np.array([[1.5, 5], [10.5, 18]], dtype=np.float32)
    run_model(converted_model, input_1, input_2, expected=[expected])


@pytest.mark.parametrize(
    ("model_filename", "inputs", "expected"),
    [
        [
            onnx_model_filename,
            [
                np.array([[1, 2], [3, 4]], dtype=np.float32),
                np.array([[2, 3], [4, 5]], dtype=np.float32),
            ],
            np.array([[1.5, 5], [10.5, 18]], dtype=np.float32),
        ],
        [
            onnx_model_with_subgraphs_filename,
            [
                np.array(False, dtype=bool),
                np.array([1, 2, 3], dtype=np.float32),
                np.array([2, 3, 5], dtype=np.float32),
            ],
            np.array([-1, -1, -2], dtype=np.float32),
        ],
    ],
)
def test_decode_and_convert(model_filename, inputs, expected):
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    assert fe

    model = fe.load(model_filename)
    assert model

    decoded_model = fe.decode(model)
    assert decoded_model

    for op in decoded_model.get_ordered_ops():
        assert op.get_type_name() in [
            "Parameter",
            "Constant",
            "ONNXFrameworkNode",
            "ONNXSubgraphFrameworkNode",
            "Result",
        ]

    fe.convert(decoded_model)
    assert decoded_model
    for op in decoded_model.get_ordered_ops():
        assert op.get_type_name() not in [
            "ONNXFrameworkNode",
            "ONNXSubgraphFrameworkNode",
        ]

    run_model(decoded_model, *inputs, expected=[expected])


def test_load_by_model():
    skip_if_onnx_frontend_is_disabled()

    fe = fem.load_by_model(onnx_model_filename)
    assert fe
    assert fe.get_name() == "onnx"
    model = fe.load(onnx_model_filename)
    assert model
    decoded_function = fe.decode(model)
    assert decoded_function

    with pytest.raises(Exception) as e:
        fem.load_by_model("test.xx")

    assert e.match(r'Could not open the file: "test.xx"')

    with pytest.raises(Exception) as e:
        fem.load_by_model("onnx.yy")

    assert e.match(r'Could not open the file: "onnx.yy"')


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
        check_attribute(node, "attribute_f32", float, 10.0)
        check_attribute(node, "attribute_f64", float, 10.0)
        check_attribute(node, "attribute_bool", int, 1)
        check_attribute(node, "attribute_type", int, 6)

        check_attribute(node, "attribute_list_i32", list, [1, 2, 3])
        check_attribute(node, "attribute_list_i64", list, [1, 2, 3])
        check_attribute(node, "attribute_list_str", list, ["a", "b", "c"])
        check_attribute(node, "attribute_list_f32", list, [1.0, 2.0, 3.0])
        check_attribute(node, "attribute_list_f64", list, [1.0, 2.0, 3.0])
        check_attribute(node, "attribute_list_bool", list, [1, 0, 1])
        check_attribute(node, "attribute_list_type", list, [6, 1])

        input_1 = node.get_input(0)
        input_2 = node.get_input(1)
        add = ops.add(input_1, input_2)
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
        check_attribute(node, "attribute_bool", False)
        check_attribute(node, "attribute_type", onnx.TensorProto.FLOAT)

        check_attribute(node, "attribute_list_i32", np.array([4, 5, 6], dtype=np.int32))
        check_attribute(node, "attribute_list_i64", np.array([4, 5, 6], dtype=np.int64))
        check_attribute(
            node, "attribute_list_str", np.array(["d", "e", "f"], dtype=str)
        )
        check_attribute(node, "attribute_list_f32", np.array([4, 5, 6], dtype=float))
        check_attribute(
            node, "attribute_list_f64", np.array([4, 5, 6], dtype=np.float64)
        )
        check_attribute(
            node, "attribute_list_bool", np.array([True, False, True], dtype=bool)
        )
        check_attribute(
            node,
            "attribute_list_type",
            np.array([onnx.TensorProto.INT32, onnx.TensorProto.FLOAT]),
        )

        input_1 = node.get_input(0)
        input_2 = node.get_input(1)
        add = ops.add(input_1, input_2)
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
        check_attribute(node, "attribute_str", "string", str)
        check_attribute(node, "attribute_f32", 10, int)
        check_attribute(node, "attribute_f64", 10, int)
        check_attribute(node, "attribute_bool", True, bool)
        check_attribute(node, "attribute_type", Type.i32, Type)

        check_attribute(node, "attribute_list_i32", [1.0, 2.0, 3.0], float)
        check_attribute(node, "attribute_list_i64", [1.0, 2.0, 3.0], float)
        check_attribute(node, "attribute_list_str", ["a", "b", "c"], str)
        check_attribute(node, "attribute_list_f32", [1, 2, 3], int)
        check_attribute(node, "attribute_list_f64", [1, 2, 3], int)
        check_attribute(node, "attribute_list_bool", [True, False, True], bool)
        check_attribute(node, "attribute_list_type", [Type.i32, Type.f32], Type)

        input_1 = node.get_input(0)
        input_2 = node.get_input(1)
        add = ops.add(input_1, input_2)
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
        input_1 = node.get_input(0)
        input_2 = node.get_input(1)
        add = ops.add(input_1, input_2)
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
        input_1 = node.get_input(0)
        input_2 = node.get_input(1)
        add = ops.add(input_1, input_2)
        return [add.output(0)]

    fe.add_extension(ConversionExtension("Add", custom_converter))
    input_model = fe.load(onnx_model_filename)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_onnx_conversion_extension_with_custom_domain():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import ConversionExtension
    from openvino.frontend import NodeContext
    import openvino.runtime.opset8 as ops

    fe = fem.load_by_model(onnx_model_extension_with_custom_domain)
    assert fe
    assert fe.get_name() == "onnx"

    invoked = False

    def custom_converter(node: NodeContext):
        nonlocal invoked
        invoked = True
        input_1 = node.get_input(0)
        input_2 = node.get_input(1)
        add = ops.add(input_1, input_2)
        return [add.output(0)]

    fe.add_extension(
        ConversionExtension("CustomAdd", "custom_domain", custom_converter)
    )
    input_model = fe.load(onnx_model_extension_with_custom_domain)
    assert input_model
    model = fe.convert(input_model)
    assert model
    assert invoked


def test_onnx_op_extension_with_custom_domain():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import OpExtension

    fe = fem.load_by_model(onnx_model_extension_with_custom_domain)
    assert fe
    assert fe.get_name() == "onnx"

    fe.add_extension(
        OpExtension(
            "opset1.Add", "CustomAdd", "custom_domain", {}, {"auto_broadcast": "numpy"}
        )
    )
    input_model = fe.load(onnx_model_extension_with_custom_domain)
    assert input_model
    model = fe.convert(input_model)
    assert model


@pytest.mark.parametrize(
    "opset_prefix", ["opset1.", "opset1::", "opset8.", "opset8::", ""]
)
def test_op_extension_specify_opset(opset_prefix):
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import OpExtension
    from openvino.runtime import Core

    core = Core()

    # check the model is valid
    model = core.read_model(onnx_model_for_op_extension_test)
    assert model

    # add extensions
    fw_operation = "Floor"
    ov_operation = opset_prefix + fw_operation
    core.add_extension(OpExtension(ov_operation, fw_operation))

    model = core.read_model(onnx_model_for_op_extension_test)
    assert model


@pytest.mark.parametrize(
    "opset_prefix", ["opset1..", "opset1:::", "opset.", "opset::", "wrong"]
)
def test_op_extension_specify_wrong_opset(opset_prefix):
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import OpExtension
    from openvino.runtime import Core

    core = Core()

    # add extensions
    fw_operation = "Floor"
    ov_operation = opset_prefix + fw_operation
    core.add_extension(OpExtension(ov_operation, fw_operation))
    with pytest.raises(RuntimeError):
        core.read_model(onnx_model_for_op_extension_test)


def test_op_extension_via_onnx_extension_set_attrs_values():
    skip_if_onnx_frontend_is_disabled()

    # use specific (openvino.frontend.onnx) import here
    from openvino.frontend.onnx import OpExtension
    from openvino.runtime import Core

    core = Core()

    # check the model is valid
    model = core.read_model(onnx_model_for_op_extension_test)
    assert model

    # add extensions
    core.add_extension(OpExtension("Multiply", "Mul", {}, {"auto_broadcast": "numpy"}))
    core.add_extension(OpExtension("Elu", {}, {"alpha": 1.0}))
    core.add_extension(OpExtension("Floor"))
    core.add_extension(OpExtension("Concat", {}, {"axis": 0}))
    core.add_extension(OpExtension("Convert", "Cast", {}, {"destination_type": "i64"}))
    core.add_extension(
        OpExtension(
            "AvgPool",
            "AveragePool",
            {},
            {
                "kernel": [2, 2],
                "strides": [2, 2],
                "pads_begin": [0, 0],
                "pads_end": [1, 1],
                "exclude-pad": True,
                "auto_pad": "same_upper",
                "rounding_type": "floor",
            },
        )
    )

    model = core.read_model(onnx_model_for_op_extension_test)
    assert model


def test_op_extension_via_frontend_extension_set_attrs_values():
    skip_if_onnx_frontend_is_disabled()

    # use common (openvino.frontend) import here
    from openvino.frontend import OpExtension
    from openvino.runtime import Core

    core = Core()
    # check the model is valid
    model = core.read_model(onnx_model_for_op_extension_test)
    assert model

    # add extensions
    core.add_extension(OpExtension("Multiply", "Mul", {}, {"auto_broadcast": "numpy"}))
    core.add_extension(OpExtension("Elu", "Elu", {}, {"alpha": 1.0}))
    core.add_extension(OpExtension("Floor"))
    core.add_extension(OpExtension("Concat", {}, {"axis": 0}))
    core.add_extension(OpExtension("Convert", "Cast", {}, {"destination_type": "i64"}))
    core.add_extension(
        OpExtension(
            "AvgPool",
            "AveragePool",
            {},
            {
                "kernel": [2, 2],
                "strides": [2, 2],
                "pads_begin": [0, 0],
                "pads_end": [1, 1],
                "exclude-pad": True,
                "auto_pad": "same_upper",
                "rounding_type": "floor",
            },
        )
    )

    model = core.read_model(onnx_model_for_op_extension_test)
    assert model


def test_op_extension_via_frontend_extension_map_attributes():
    skip_if_onnx_frontend_is_disabled()

    # use common (openvino.frontend) import here
    from openvino.frontend import OpExtension
    from openvino.runtime import Core

    core = Core()
    # check the model is valid
    model = core.read_model(onnx_model_for_op_extension_test)
    assert model

    # add extensions
    core.add_extension(OpExtension("Elu", "Elu", {"alpha": "alpha"}))
    core.add_extension(OpExtension("Concat", {"axis": "axis"}, {"axis": 0}))

    core.add_extension(
        OpExtension(
            "AvgPool",
            "AveragePool",
            {"kernel": "kernel_shape", "strides": "strides", "auto_pad": "auto_pad"},
            {
                "pads_begin": [0, 0],
                "pads_end": [1, 1],
                "exclude-pad": True,
                "rounding_type": "floor",
            },
        )
    )

    model = core.read_model(onnx_model_for_op_extension_test)
    assert model


def get_builtin_extensions_path():
    ci_tests_path = Path(__file__).resolve().parents[3]
    for lib_path in chain(
        ci_tests_path.glob("*.dll"), ci_tests_path.glob("*.so")
    ):
        if "test_builtin_extensions" in lib_path.name:
            return str(lib_path)
    return ""


@pytest.mark.skipif(
    len(get_builtin_extensions_path()) == 0,
    reason="The extension library path was not found",
)
def test_so_extension_via_frontend_convert_input_model():
    skip_if_onnx_frontend_is_disabled()

    def load_model():
        fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
        fe.add_extension(get_builtin_extensions_path())
        in_model = fe.load(onnx_model_2_filename)
        return fe.convert(in_model)

    model = load_model()  # model has longer lifetime than frontend

    assert any(op.get_type_name() == "Swish" for op in model.get_ops())
    assert all(op.get_type_name() != "Relu" for op in model.get_ops())


@pytest.mark.skipif(
    len(get_builtin_extensions_path()) == 0,
    reason="The extension library path was not found",
)
def test_so_extension_via_frontend_decode_input_model():
    skip_if_onnx_frontend_is_disabled()

    def load_decoded_model():
        fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
        fe.add_extension(get_builtin_extensions_path())
        in_model = fe.load(onnx_model_2_filename)
        return fe.decode(in_model)

    decoded_model = (
        load_decoded_model()
    )  # decoded model has longer lifetime than frontend
    assert decoded_model


@pytest.mark.skipif(
    len(get_builtin_extensions_path()) == 0,
    reason="The extension library path was not found",
)
def test_add_extension_unicode_paths():
    skip_if_onnx_frontend_is_disabled()

    test_directory = Path(__file__).resolve().parent
    unicode_characters = r"晚安_путь_к_файлу"
    with tempfile.TemporaryDirectory(dir=test_directory, prefix=unicode_characters) as temp_dir:
        extension_path = Path(get_builtin_extensions_path())
        temp_extension_path = Path(temp_dir) / extension_path.name
        shutil.copyfile(extension_path, temp_extension_path)

        assert os.path.exists(temp_extension_path), "Could not create an extension library with unicode path."

        def convert_model(path):
            fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
            fe.add_extension(path)
            in_model = fe.load(onnx_model_2_filename)
            converted_model = fe.convert(in_model)
            assert converted_model

        convert_model(temp_extension_path)


def test_load_bytesio_model():
    from openvino.runtime import Core

    fe = fem.load_by_framework(framework=ONNX_FRONTEND_NAME)
    model_from_fe = fe.load(model_stream)
    assert model_from_fe
    converted_model = fe.convert(model_from_fe)
    assert converted_model.friendly_name == "graph"

    core = Core()
    model = core.read_model(model_stream)
    assert converted_model.friendly_name == model.friendly_name
