# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime.opset8 as ov
from openvino.runtime.exceptions import UserInputError
from openvino.runtime.utils.node_factory import NodeFactory


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
    k = ov.constant(3, dtype=dtype, name="B")
    factory = NodeFactory("opset1")
    arguments = NodeFactory._arguments_as_outputs([data, k])
    node = factory.create(
        "TopK", arguments, {"axis": 1, "mode": "max", "sort": "value"}
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
    k = ov.constant(3, dtype=dtype, name="B")
    factory = NodeFactory("opset1")
    arguments = NodeFactory._arguments_as_outputs([data, k])
    node = factory.create("TopK", None, None)
    node.set_arguments(arguments)
    node.set_attribute("axis", 1)
    node.set_attribute("mode", "max")
    node.set_attribute("sort", "value")
    node.validate()

    assert node.get_type_name() == "TopK"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [2, 3]


def test_node_factory_validate_missing_arguments():
    factory = NodeFactory("opset1")

    try:
        factory.create(
            "TopK", None, {"axis": 1, "mode": "max", "sort": "value"}
        )
    except UserInputError:
        pass
    else:
        raise AssertionError("Validation of missing arguments has unexpectedly passed.")
