# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import numpy as np
import pytest
from sys import platform
from openvino import compile_model, Model
from openvino import DiscreteTypeInfo, Extension, Op, OpExtension, Tensor, Type
import openvino.opset8 as ov
from openvino.exceptions import UserInputError
from openvino.utils.node_factory import NodeFactory


def test_node_factory_add():
    shape = [2, 2]
    dtype = np.int8
    parameter_a = ov.parameter(shape, dtype=dtype, name="A")
    parameter_b = ov.parameter(shape, dtype=dtype, name="B")

    factory = NodeFactory("opset1")
    arguments = NodeFactory._arguments_as_outputs([parameter_a, parameter_b])
    node = factory.create("Add", arguments, {})

    assert node.get_type_name() == "Add"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]


def test_node_factory_wrapper_add():
    shape = [2, 2]
    dtype = np.int8
    parameter_a = ov.parameter(shape, dtype=dtype, name="A")
    parameter_b = ov.parameter(shape, dtype=dtype, name="B")

    node = ov.add(parameter_a, parameter_b, name="TestNode")

    assert node.get_type_name() == "Add"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]
    assert node.friendly_name == "TestNode"


def test_node_factory_topk():
    dtype = np.int32
    data = ov.parameter([2, 10], dtype=dtype, name="A")
    const = ov.constant(3, dtype=dtype, name="B")
    factory = NodeFactory("opset1")
    arguments = NodeFactory._arguments_as_outputs([data, const])
    node = factory.create(
        "TopK", arguments, {"axis": 1, "mode": "max", "sort": "value"},
    )
    attributes = node.get_attributes()

    assert node.get_type_name() == "TopK"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [2, 3]
    assert attributes["axis"] == 1
    assert attributes["mode"] == "max"
    assert attributes["sort"] == "value"


def test_node_factory_empty_topk():
    factory = NodeFactory("opset1")
    node = factory.create("TopK")

    assert node.get_type_name() == "TopK"


def test_node_factory_empty_topk_with_args_and_attrs():
    dtype = np.int32
    data = ov.parameter([2, 10], dtype=dtype, name="A")
    const = ov.constant(3, dtype=dtype, name="B")
    factory = NodeFactory("opset1")
    arguments = NodeFactory._arguments_as_outputs([data, const])
    node = factory.create("TopK", None, None)
    node.set_arguments(arguments)
    node.set_attribute("axis", 1)
    node.set_attribute("mode", "max")
    node.set_attribute("sort", "value")

    node.constructor_validate_and_infer_types()

    assert node.get_type_name() == "TopK"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [2, 3]


def test_node_factory_validate_missing_arguments():
    factory = NodeFactory("opset1")

    try:
        factory.create(
            "TopK", None, {"axis": 1, "mode": "max", "sort": "value"},
        )
    except UserInputError:
        pass
    else:
        raise AssertionError("Validation of missing arguments has unexpectedly passed.")


@pytest.mark.template_extension
@pytest.mark.dynamic_library
@pytest.mark.xfail(condition=platform == "darwin", reason="Ticket - 132696")
def test_extension_added_from_library():
    if platform == "win32":
        library_path = "openvino_template_extension.dll"
    else:
        library_path = "libopenvino_template_extension.so"

    factory = NodeFactory()
    factory.add_extension(library_path)

    data = ov.parameter([1, 2], dtype=np.float32)
    identity = factory.create("Identity", data.outputs())
    model = Model([identity], [data])
    compiled = compile_model(model)
    tensor = np.array([[3, 4]], dtype=np.float32)
    result = compiled(tensor)

    # TODO: There is an issue with life time of objects, free resources explicitly
    # otherwise segfault will occur. Workaround: create factory as a global variable.
    del compiled
    del model
    del identity

    assert np.array_equal(tensor, result[0])


def test_add_extension():
    class EmptyExtension(Extension):
        def __init__(self) -> None:
            super().__init__()

    factory = NodeFactory()
    factory.add_extension(EmptyExtension())
    factory.add_extension([EmptyExtension(), EmptyExtension()])

    data = ov.parameter([1, 2], dtype=np.float32)
    param = factory.create("Parameter", data.outputs())
    assert param is not None


@pytest.mark.parametrize("registration", ["direct", "list"])
def test_plain_extension_retains_owner(registration):
    class PlainIdentity(Op):
        class_type_info = DiscreteTypeInfo("PlainFactoryIdentity", "extension")

        def __init__(self, inputs=None):
            super().__init__(self)
            if inputs is not None:
                self.set_arguments(inputs)
                self.constructor_validate_and_infer_types()

        def get_type_info(self):
            return self.class_type_info

        def visit_attributes(self, visitor):
            return True

        def validate_and_infer_types(self):
            self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

        def clone_with_new_inputs(self, inputs):
            return PlainIdentity(inputs)

        def has_evaluate(self):
            return True

        def evaluate(self, outputs, inputs):
            inputs[0].copy_to(outputs[0])
            return True

    class EmptyExtension(Extension):
        def __init__(self):
            super().__init__()

    factory = NodeFactory()
    extension = OpExtension(PlainIdentity)
    factory.add_extension(extension if registration == "direct" else [EmptyExtension(), extension])
    del extension
    gc.collect()

    input_parameter = ov.parameter([2, 3], dtype=np.float32)
    node = factory.create("PlainFactoryIdentity", [input_parameter], {})
    assert list(node.output(0).shape) == [2, 3]
    assert node.output(0).element_type == Type.f32
    model = Model([node], [input_parameter])
    values = np.array([[1.25, -2.5, 0], [3.5, -1, 7.75]], dtype=np.float32)
    output = Tensor(Type.f32, [2, 3])
    assert model.evaluate([output], [Tensor(values)])
    np.testing.assert_array_equal(output.data, values)
    del model, node
