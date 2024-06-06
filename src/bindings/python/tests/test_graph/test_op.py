# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino import Op
from openvino.runtime import CompiledModel, DiscreteTypeInfo, Model, Shape, compile_model, Tensor
import openvino.runtime.opset14 as ops


class CustomOp(Op):
    class_type_info = DiscreteTypeInfo("Custom", "extension")

    def __init__(self, inputs):
        super().__init__(self)
        self.set_arguments(inputs)
        self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return CustomOp(new_inputs)

    def get_type_info(self):
        return CustomOp.class_type_info

    def evaluate(self, outputs, inputs):
        inputs[0].copy_to(outputs[0])
        return True

    def has_evaluate(self):
        return True


def create_snake_model():
    input_shape = [1, 3, 32, 32]
    param1 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data")
    custom_op = CustomOp([param1])
    custom_op.set_friendly_name("custom_" + str(0))

    for i in range(20):
        custom_op = CustomOp([custom_op])
        custom_op.set_friendly_name("custom_" + str(i + 1))
    return Model(custom_op, [param1], "TestModel")


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


def create_add_model():
    input_shape = [2, 1]

    param1 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data2")
    custom_add = CustomAdd(inputs=[param1, param2])
    custom_add.set_friendly_name("test_add")
    res = ops.result(custom_add, name="result")
    return Model(res, [param1, param2], "AddModel")


def test_custom_add_op():
    data1 = np.array([1, 2, 3])
    data2 = np.array([4, 5, 6])

    node1 = ops.constant(data1, dtype=np.float32)
    node2 = ops.constant(data2, dtype=np.float32)
    inputs = [node1.output(0), node2.output(0)]
    custom_op = CustomAdd(inputs=inputs)
    custom_op.set_friendly_name("test_add")

    assert custom_op.get_input_size() == 2
    assert custom_op.get_output_size() == 1
    assert custom_op.get_type_name() == "CustomAdd"
    assert list(custom_op.get_output_shape(0)) == [3]
    assert custom_op.friendly_name == "test_add"


def test_custom_add_model():
    model = create_add_model()

    assert isinstance(model, Model)

    ordered_ops = model.get_ordered_ops()
    assert len(ordered_ops) == 4

    op_types = [op.get_type_name() for op in ordered_ops]
    assert op_types == ["Parameter", "Parameter", "CustomAdd", "Result"]


def test_custom_op():
    model = create_snake_model()
    # todo: CVS-141744
    # it hangs with AUTO plugin, but works well with CPU
    compiled_model = compile_model(model, "CPU")

    assert isinstance(compiled_model, CompiledModel)
    request = compiled_model.create_infer_request()

    input_data = np.ones([1, 3, 32, 32], dtype=np.float32)
    expected_output = np.maximum(0.0, input_data)

    input_tensor = Tensor(input_data)
    results = request.infer({"data": input_tensor})
    assert np.allclose(results[list(results)[0]], expected_output, 1e-4, 1e-4)
