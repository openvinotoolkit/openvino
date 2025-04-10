# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import pytest
import numpy as np
from contextlib import nullcontext as does_not_raise

from openvino import Op, OpExtension
from openvino import CompiledModel, Core, Model, Dimension, Shape, Tensor, compile_model, serialize
from openvino import DiscreteTypeInfo
import openvino.opset14 as ops

from tests.utils.helpers import create_filenames_for_ir, compare_models


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

    def __init__(self, inputs=None):
        super().__init__(self)
        if inputs is not None:
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


class CustomOpWithAttribute(Op):
    class_type_info = DiscreteTypeInfo("CustomOpWithAttribute", "extension")

    def __init__(self, inputs=None, attrs=None):
        super().__init__(self)
        self._attrs = attrs
        if inputs is not None:
            self.set_arguments(inputs)
            self.constructor_validate_and_infer_types()

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    def clone_with_new_inputs(self, new_inputs):
        return CustomOpWithAttribute(new_inputs)

    def get_type_info(self):
        return CustomOpWithAttribute.class_type_info

    def visit_attributes(self, visitor):
        visitor.on_attributes(self._attrs)
        return True

    def evaluate(self, outputs, inputs):
        inputs[0].copy_to(outputs[0])
        return True


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
@pytest.fixture
def prepared_paths(request, tmp_path):
    xml_path, bin_path = create_filenames_for_ir(request.node.name, tmp_path)

    yield xml_path, bin_path

    os.remove(xml_path)
    os.remove(bin_path)


@pytest.mark.parametrize(("attributes", "expectation", "raise_msg"), [
    ({"axis": 0}, does_not_raise(), ""),
    ({"value_str": "test_attribute"}, does_not_raise(), ""),
    ({"value_float": 0.25}, does_not_raise(), ""),
    ({"value_bool": True}, does_not_raise(), ""),
    ({"list_str": ["one", "two"]}, does_not_raise(), ""),
    ({"list_int": [1, 2]}, does_not_raise(), ""),
    ({"list_float": np.array([1.5, 2.5], dtype="float32")}, does_not_raise(), ""),
    ({"axis": 0, "list_int": [1, 2]}, does_not_raise(), ""),
    ({"body": Model(ops.constant([1]), [])}, does_not_raise(), ""),
    ({"dim": Dimension(2)}, does_not_raise(), ""),
    ({"wrong_list": [{}]}, pytest.raises(TypeError), "Unsupported attribute type in provided list: <class 'dict'>"),
    ({"wrong_np": np.array([1.5, 2.5], dtype="complex128")}, pytest.raises(TypeError), "Unsupported NumPy array dtype: complex128"),
    ({"wrong": {}}, pytest.raises(TypeError), "Unsupported attribute type: <class 'dict'>")
])
@pytest.mark.skipif(sys.platform == "win32", reason="CVS-164354 BUG: hanged on windows wheels")
def test_visit_attributes_custom_op(prepared_paths, attributes, expectation, raise_msg):
    input_shape = [2, 1]

    param1 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data2")
    custom = CustomOpWithAttribute(inputs=[param1, param2], attrs=attributes)
    res = ops.result(custom, name="result")
    model_with_op_attr = Model(res, [param1, param2], "CustomModel")

    xml_path, bin_path = prepared_paths

    with expectation as e:
        serialize(model_with_op_attr, xml_path, bin_path)
        ordered_ops = model_with_op_attr.get_ordered_ops()
        ops_dict = {op.get_type_name(): op for op in ordered_ops}
        attrs = ops_dict["CustomOpWithAttribute"].get_attributes()
        for key, value in attrs.items():
            assert key in attributes
            assert attributes[key] == value

    if e is not None:
        assert raise_msg in str(e.value)

    input_data = np.ones([2, 1], dtype=np.float32)
    expected_output = np.maximum(0.0, input_data)

    compiled_model = compile_model(model_with_op_attr)
    input_tensor = Tensor(input_data)
    results = compiled_model({"data1": input_tensor})
    assert np.allclose(results[list(results)[0]], expected_output, 1e-4, 1e-4)


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
    compiled_model = compile_model(model)

    assert isinstance(compiled_model, CompiledModel)
    request = compiled_model.create_infer_request()

    input_data = np.ones([1, 3, 32, 32], dtype=np.float32)
    expected_output = np.maximum(0.0, input_data)

    input_tensor = Tensor(input_data)
    results = request.infer({"data": input_tensor})
    assert np.allclose(results[list(results)[0]], expected_output, 1e-4, 1e-4)


class CustomSimpleOp(Op):
    def __init__(self, *inputs):
        super().__init__(self, inputs)

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))


class CustomSimpleOpWithAttribute(Op):
    class_type_info = DiscreteTypeInfo("CustomSimpleOpWithAttribute", "extension")

    def __init__(self, inputs=None, **attrs):
        super().__init__(self, inputs)
        self._attrs = attrs

    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))

    @staticmethod
    def get_type_info():
        return CustomSimpleOpWithAttribute.class_type_info

    def visit_attributes(self, visitor):
        if self._attrs:
            visitor.on_attributes(self._attrs)
        return True


def test_op_extension(prepared_paths):
    input_shape = [2, 1]

    core = Core()
    core.add_extension(CustomSimpleOp)
    core.add_extension(OpExtension(CustomSimpleOpWithAttribute))
    core.add_extension(OpExtension(CustomAdd))

    param1 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape(input_shape), dtype=np.float32, name="data2")
    custom_simple = CustomSimpleOp(param1, param2)
    assert custom_simple.get_type_name() == "CustomSimpleOp"
    custom_simple.set_friendly_name("test_add")
    custom_with_attribute = CustomSimpleOpWithAttribute([custom_simple], value_str="test_attribute")
    custom_add = CustomAdd(inputs=[custom_with_attribute])
    res = ops.result(custom_add, name="result")
    simple_model = Model(res, [param1, param2], "SimpleModel")

    cloned_model = simple_model.clone()
    assert compare_models(simple_model, cloned_model)

    xml_path, bin_path = prepared_paths
    serialize(simple_model, xml_path, bin_path)
    model_with_custom_op = core.read_model(xml_path)
    assert compare_models(simple_model, model_with_custom_op)


def test_fail_create_op_extension():
    class OpWithBadClassTypeInfo(Op):
        class_type_info = "OpWithBadClassTypeInfo"

        def __init__(self, inputs=None):
            super().__init__(self)
            if inputs is not None:
                self.set_arguments(inputs)
                self.constructor_validate_and_infer_types()

        @staticmethod
        def get_type_info():
            return OpWithBadClassTypeInfo.class_type_info

    core = Core()
    with pytest.raises(RuntimeError) as e:
        core.add_extension(CustomOp)
    assert "__init__() missing 1 required positional argument: 'inputs'" in str(e.value)

    with pytest.raises(RuntimeError) as e:
        core.add_extension(OpExtension(bool))
    assert "Unsupported data type : '<class 'bool'>' is passed as an argument." in str(e.value)

    with pytest.raises(RuntimeError) as e:
        core.add_extension(bool)
    assert "Unsupported data type : '<class 'bool'>' is passed as an argument." in str(e.value)

    with pytest.raises(RuntimeError) as e:
        core.add_extension(OpWithBadClassTypeInfo)
    assert "operation type_info must be an instance of DiscreteTypeInfo, but <class 'str'> is passed." in str(e.value)
