# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino import Op
from openvino import DiscreteTypeInfo, Shape
import openvino.opset14 as ops
from openvino.utils.node_factory import NodeFactory


class CustomAdd(Op):
    class_type_info = DiscreteTypeInfo("CustomAdd", "extension")

    def __init__(self, inputs):
        super().__init__(self)
        self.set_arguments(inputs)
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        node = CustomAdd(new_inputs)
        return node

    def get_type_info(self):
        return CustomAdd.class_type_info


def test_node_factory_type_info():
    shape = [2, 2]
    dtype = np.int8
    parameter_a = ops.parameter(shape, dtype=dtype, name="A")
    parameter_b = ops.parameter(shape, dtype=dtype, name="B")

    factory = NodeFactory("opset1")
    arguments = NodeFactory._arguments_as_outputs([parameter_a, parameter_b])
    node = factory.create("Add", arguments, {})

    type_info = node.get_type_info()
    assert isinstance(type_info, DiscreteTypeInfo)
    assert type_info.name == "Add"
    assert type_info.version_id == "opset1"


def test_discrete_type_info():
    type_info = DiscreteTypeInfo("Custom", "extension")
    assert isinstance(type_info, DiscreteTypeInfo)
    assert type_info.name == "Custom"
    assert type_info.version_id == "extension"


def test_custom_add_op_type_info():
    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])

    node1 = ops.constant(data1, dtype=np.float32)
    node2 = ops.constant(data2, dtype=np.float32)
    inputs = [node1.output(0), node2.output(0)]
    custom_op = CustomAdd(inputs=inputs)

    type_info = custom_op.get_type_info()
    assert isinstance(type_info, DiscreteTypeInfo)
    assert type_info.name == "CustomAdd"
    assert type_info.version_id == "extension"


def test_add_type_info():
    input_shape = [2, 1]
    param1 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)

    type_info = add.get_type_info()
    assert isinstance(type_info, DiscreteTypeInfo)
    assert type_info.name == "Add"
    assert type_info.version_id == "opset1"
