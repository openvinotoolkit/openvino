# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pytest
import math
from contextlib import nullcontext as does_not_raise
from copy import copy

import openvino.runtime.opset13 as ops
from openvino import (
    Core,
    Model,
    Tensor,
    Dimension,
    Layout,
    Type,
    PartialShape,
    Shape,
    set_batch,
    get_batch,
    serialize,
    save_model,
)
from openvino.runtime import Output
from openvino.runtime.op.util import VariableInfo, Variable

from tests.utils.helpers import (
    generate_add_model,
    generate_model_with_memory,
    create_filename_for_test,
)


def make_add_with_variable_model(shape, variable_id: str, dtype=np.float32) -> Model:
    param1 = ops.parameter(Shape(shape), dtype=dtype, name="data1")
    param2 = ops.parameter(Shape(shape), dtype=dtype, name="data2")
    read_value = ops.read_value(param1, variable_id, dtype, shape)
    return Model(ops.add(read_value, param2), [param1, param2])


def test_descriptor_tensor():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    td = relu1.get_output_tensor(0)
    assert "relu_t1" in td.names
    assert td.element_type == Type.f32
    assert td.partial_shape == PartialShape([1])
    assert repr(td.shape) == "<Shape: [1]>"
    assert td.size == 4
    assert td.any_name == "relu_t1"


@pytest.mark.parametrize(("output", "expectation", "raise_msg"), [
    ("relu_t1", does_not_raise(), ""),
    (("relu1", 0), does_not_raise(), ""),
    ("relu_t", pytest.raises(RuntimeError), "relu_t"),
    (("relu1", 1234), pytest.raises(RuntimeError), "1234"),
    (("relu_1", 0), pytest.raises(RuntimeError), "relu_1"),
    (0, pytest.raises(TypeError), "Incorrect type of a value to add as output."),
    ([0, 0], pytest.raises(TypeError), "Incorrect type of a value to add as output at index 0"),
])
def test_add_outputs(output, expectation, raise_msg):
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    assert "relu_t1" in relu1.get_output_tensor(0).names
    relu2 = ops.relu(relu1, name="relu2")
    model = Model(relu2, [param], "TestModel")
    assert len(model.get_results()) == 1
    assert len(model.results) == 1
    with expectation as e:
        new_outs = model.add_outputs(output)
        assert len(model.get_results()) == 2
        assert len(model.results) == 2
        assert "relu_t1" in model.outputs[1].get_tensor().names
        assert len(new_outs) == 1
        assert new_outs[0].get_node().get_instance_id() == model.outputs[1].get_node().get_instance_id()
        assert new_outs[0].get_index() == model.outputs[1].get_index()
    if e is not None:
        assert raise_msg in str(e.value)


def test_add_output_port():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    model = Model(relu2, [param], "TestModel")
    assert len(model.results) == 1
    new_outs = model.add_outputs(relu1.output(0))
    assert len(model.results) == 2
    assert len(new_outs) == 1
    assert new_outs[0].get_node().get_instance_id() == model.outputs[1].get_node().get_instance_id()
    assert new_outs[0].get_index() == model.outputs[1].get_index()


@pytest.mark.parametrize("args", [["relu_t1", "relu_t2"], [("relu1", 0), ("relu2", 0)]])
def test_add_outputs_several_outputs(args):
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    relu2.get_output_tensor(0).set_names({"relu_t2"})
    relu3 = ops.relu(relu2, name="relu3")
    model = Model(relu3, [param], "TestModel")
    assert len(model.get_results()) == 1
    assert len(model.results) == 1
    new_outs = model.add_outputs(args)
    assert len(model.get_results()) == 3
    assert len(model.results) == 3
    assert len(new_outs) == 2
    assert new_outs[0].get_node().get_instance_id() == model.outputs[1].get_node().get_instance_id()
    assert new_outs[0].get_index() == model.outputs[1].get_index()
    assert new_outs[1].get_node().get_instance_id() == model.outputs[2].get_node().get_instance_id()
    assert new_outs[1].get_index() == model.outputs[2].get_index()


def test_validate_nodes_and_infer_types():
    model = generate_add_model()
    invalid_shape = Shape([3, 7])
    param3 = ops.parameter(invalid_shape, dtype=np.float32, name="data3")
    model.replace_parameter(0, param3)

    with pytest.raises(RuntimeError) as e:
        model.validate_nodes_and_infer_types()
    assert "Argument shapes are inconsistent" in str(e.value)


def test_get_result_index():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu = ops.relu(param, name="relu")
    model = Model(relu, [param], "TestModel")
    assert len(model.outputs) == 1
    assert model.get_result_index(model.outputs[0]) == 0


def test_get_result_index_invalid():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data1")
    relu1 = ops.relu(param1, name="relu1")
    model = Model(relu1, [param1], "TestModel")

    shape2 = PartialShape([2])
    param2 = ops.parameter(shape2, dtype=np.float32, name="data2")
    relu2 = ops.relu(param2, name="relu2")
    invalid_output = relu2.outputs()[0]
    assert len(model.outputs) == 1
    assert model.get_result_index(invalid_output) == -1


@pytest.mark.parametrize(("shapes", "relu_names", "model_name", "expected_outputs_length", "is_invalid", "expected_result_index"), [
    ([PartialShape([1])], ["relu"], "TestModel", 1, False, 0),
    ([PartialShape([1]), PartialShape([4])], ["relu1", "relu2"], "TestModel1", 1, True, -1)
])
def test_result_index(shapes, relu_names, model_name, expected_outputs_length, is_invalid, expected_result_index):
    params = [ops.parameter(shape, dtype=np.float32, name=f"data{i+1}") for i, shape in enumerate(shapes)]
    relus = [ops.relu(param, name=relu_name) for param, relu_name in zip(params, relu_names)]

    model = Model(relus[0], [params[0]], model_name)
    assert len(model.outputs) == expected_outputs_length
    if is_invalid:
        invalid_result_node = ops.result(relus[1].outputs()[0])
        assert model.get_result_index(invalid_result_node) == expected_result_index
    else:
        assert model.get_result_index(model.get_results()[0]) == expected_result_index


@pytest.mark.parametrize(("shapes", "param_names", "model_name", "expected_index", "is_invalid"), [
    ([PartialShape([1]), None], ["data", None], "TestModel", 0, False),
    ([PartialShape([1]), PartialShape([2])], ["data1", "data2"], "TestModel", -1, True)
])
def test_parameter_index(shapes, param_names, model_name, expected_index, is_invalid):
    param1 = ops.parameter(shapes[0], dtype=np.float32, name=param_names[0])
    relu = ops.relu(param1, name="relu")
    model = Model(relu, [param1], model_name)

    if is_invalid:
        param2 = ops.parameter(shapes[1], dtype=np.float32, name=param_names[1])
        assert model.get_parameter_index(param2) == expected_index
    else:
        assert model.get_parameter_index(param1) == expected_index


def test_replace_parameter():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data")
    shape2 = PartialShape([2])
    param2 = ops.parameter(shape2, dtype=np.float32, name="data")
    relu = ops.relu(param1, name="relu")

    model = Model(relu, [param1], "TestModel")
    param_index = model.get_parameter_index(param1)
    model.replace_parameter(param_index, param2)
    assert model.get_parameter_index(param2) == param_index
    assert model.get_parameter_index(param1) == -1


def test_get_sink_index(device):
    input_shape = PartialShape([2, 2])
    param = ops.parameter(input_shape, dtype=np.float64, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    model = Model(relu1, [param], "TestModel")

    # test get_sink_index with openvino.runtime.Node argument
    assign = ops.assign()
    assign2 = ops.assign()
    assign3 = ops.assign()
    model.add_sinks([assign, assign2, assign3])
    assign_nodes = model.sinks
    assert model.get_sink_index(assign_nodes[2]) == 2
    assert model.get_sink_index(relu1) == -1

    # test get_sink_index with openvino.runtime.Output argument
    assign4 = ops.assign(relu1, "assign4")
    model.add_sinks([assign4])
    assert model.get_sink_index(assign4.output(0)) == 3

    # check exceptions
    with pytest.raises(TypeError) as e:
        model.get_sink_index(0)
    assert (
        "Incorrect argument type. Sink node is expected as argument." in str(e.value)
    )


def test_model_sink_ctors():
    input_data = ops.parameter([2, 2], name="input_data", dtype=np.float32)
    rv = ops.read_value("var_id_667", np.float32, [2, 2])
    add = ops.add(rv, input_data, name="MemoryAdd")
    node = ops.assign(add, "var_id_667")
    res = ops.result(add, "res")

    # Model(List[openvino._pyopenvino.op.Result], List[ov::Output<ov::Node>],
    # List[openvino._pyopenvino.op.Parameter], str = '')
    model = Model(results=[res], sinks=[node.output(0)], parameters=[input_data], name="TestModel")
    model.validate_nodes_and_infer_types()
    sinks = ["Assign"]
    assert sinks == [sink.get_type_name() for sink in model.get_sinks()]
    assert model.sinks[0].get_output_shape(0) == Shape([2, 2])

    # Model(List[ov::Output<ov::Node>, List[ov::Output<ov::Node>],
    # List[openvino._pyopenvino.op.Parameter], str = '')
    model = Model(results=[res.output(0)], sinks=[node.output(0)], parameters=[input_data], name="TestModel")
    model.validate_nodes_and_infer_types()
    assert model.sinks[0].get_output_shape(0) == Shape([2, 2])
    assert sinks == [sink.get_type_name() for sink in model.get_sinks()]

    var_info = VariableInfo()
    var_info.data_shape = PartialShape([2, 2])
    var_info.data_type = Type.f32
    var_info.variable_id = "v1"
    variable_1 = Variable(var_info)
    rv = ops.read_value(variable_1)
    add = ops.add(rv, input_data, name="MemoryAdd")
    assign = ops.assign(add, variable_1)
    res = ops.result(add, "res")

    # Model(List[openvino._pyopenvino.op.Result], List[ov::Output<ov::Node>],
    # List[openvino._pyopenvino.op.Parameter], List[openvino._pyopenvino.op.util.Variable], str = '')
    model = Model(results=[res], sinks=[assign.output(0)], parameters=[input_data], variables=[variable_1], name="TestModel")
    model.validate_nodes_and_infer_types()
    assert model.sinks[0].get_output_shape(0) == Shape([2, 2])
    assert sinks == [sink.get_type_name() for sink in model.get_sinks()]

    # Model(List[ov::Output<ov::Node>, List[ov::Output<ov::Node>],
    # List[openvino._pyopenvino.op.Parameter], List[openvino._pyopenvino.op.util.Variable], str = '')
    model = Model(results=[res.output(0)], sinks=[assign.output(0)], parameters=[input_data], variables=[variable_1], name="TestModel")
    model.validate_nodes_and_infer_types()
    assert model.sinks[0].get_output_shape(0) == Shape([2, 2])
    assert sinks == [sink.get_type_name() for sink in model.get_sinks()]


@pytest.mark.parametrize(("args1", "args2", "expectation", "raise_msg"), [
    (Tensor("float32", Shape([2, 1])),
     [Tensor(np.array([2, 1], dtype=np.float32).reshape(2, 1)),
      Tensor(np.array([3, 7], dtype=np.float32).reshape(2, 1))], does_not_raise(), ""),
    (Tensor("float32", Shape([2, 1])),
     [Tensor("float32", Shape([3, 1])),
      Tensor("float32", Shape([3, 1]))], pytest.raises(RuntimeError), "Cannot evaluate model!"),
])
def test_evaluate(args1, args2, expectation, raise_msg):
    model = generate_add_model()
    with expectation as e:
        out_tensor = args1
        assert model.evaluate([out_tensor], args2)
        assert np.allclose(out_tensor.data, np.array([5, 8]).reshape(2, 1))
    if e is not None:
        assert raise_msg in str(e.value)


def test_get_batch():
    model = generate_add_model()
    param = model.get_parameters()[0]
    param.set_layout(Layout("NC"))
    assert get_batch(model) == 2
    param = model.parameters[0]
    param.set_layout(Layout("NC"))
    assert get_batch(model) == 2


def test_get_batch_chwn():
    param1 = ops.parameter(Shape([3, 1, 3, 4]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([3, 1, 3, 4]), dtype=np.float32, name="data2")
    param3 = ops.parameter(Shape([3, 1, 3, 4]), dtype=np.float32, name="data3")
    add = ops.add(param1, param2)
    add2 = ops.add(add, param3)
    model = Model(add2, [param1, param2, param3], "TestModel")
    param_method = model.get_parameters()[0]
    param_attr = model.parameters[0]
    param_method.set_layout(Layout("CHWN"))
    param_attr.set_layout(Layout("CHWN"))
    assert get_batch(model) == 4


@pytest.mark.parametrize("batch_arg", [Dimension(1), 1])
def test_set_batch(batch_arg):
    model = generate_add_model()
    model_param1_method = model.get_parameters()[0]
    model_param2_method = model.get_parameters()[1]
    model_param1_attr = model.parameters[0]
    model_param2_attr = model.parameters[1]
    # check batch == 2
    model_param1_method.set_layout(Layout("NC"))
    model_param1_attr.set_layout(Layout("NC"))
    assert get_batch(model) == 2
    # set batch to 1
    set_batch(model, batch_arg)
    assert get_batch(model) == 1
    # check if shape of param 1 has changed
    assert model_param1_method.get_output_shape(0) == PartialShape([1, 1])
    assert model_param1_attr.get_output_shape(0) == PartialShape([1, 1])
    # check if shape of param 2 has not changed
    assert model_param2_method.get_output_shape(0) == PartialShape([2, 1])
    assert model_param2_attr.get_output_shape(0) == PartialShape([2, 1])


def test_set_batch_default_batch_size():
    model = generate_add_model()
    model_param1 = model.get_parameters()[0]
    model_param1.set_layout(Layout("NC"))
    set_batch(model)
    assert model.is_dynamic()
    assert model.dynamic


def test_reshape_with_ports():
    model = generate_add_model()
    new_shape = PartialShape([1, 4])
    for model_input in model.inputs:
        assert isinstance(model_input, Output)
        model.reshape({model_input: new_shape})
        assert model_input.partial_shape == new_shape


def test_reshape_with_indexes():
    model = generate_add_model()
    new_shape = PartialShape([1, 4])
    for index, model_input in enumerate(model.inputs):
        model.reshape({index: new_shape})
        assert model_input.partial_shape == new_shape


def test_reshape_with_names():
    model = generate_add_model()
    new_shape = PartialShape([1, 4])
    for model_input in model.inputs:
        model.reshape({model_input.any_name: new_shape})
        assert model_input.partial_shape == new_shape


def test_reshape(device):
    shape = Shape([1, 10])
    param = ops.parameter(shape, dtype=np.float32)
    model = Model(ops.relu(param), [param])
    ref_shape = model.input().partial_shape
    ref_shape[0] = 3
    model.reshape(ref_shape)
    core = Core()
    compiled_model = core.compile_model(model, device)
    assert compiled_model.input().partial_shape == ref_shape


def test_reshape_with_ports_and_variable():
    new_shape = PartialShape([46, 1])
    var_id = "ID1"
    model = make_add_with_variable_model([1, 5], var_id)
    for model_input in model.inputs:
        assert isinstance(model_input, Output)
        model.reshape({model_input: new_shape}, {var_id: new_shape})
        assert model_input.partial_shape == new_shape


def test_reshape_with_indexes_and_variable():
    var_id = "ID1"
    model = make_add_with_variable_model([1, 5], var_id)
    new_shape = PartialShape([1, 4])
    model.reshape(
        {i: new_shape for i, model_input in enumerate(model.inputs)},
        {var_id: new_shape},
    )
    for model_input in model.inputs:
        assert model_input.partial_shape == new_shape


def test_reshape_with_names_and_variables():
    var_id = "ID1"
    model = make_add_with_variable_model([1, 25], var_id)
    new_shape = PartialShape([4, 1])
    model.reshape(
        {model_input.any_name: new_shape for model_input in model.inputs},
        {var_id: new_shape},
    )
    for model_input in model.inputs:
        assert model_input.partial_shape == new_shape


def test_reshape_with_variable(device):
    shape = PartialShape([1, 10])

    param = ops.parameter(shape, dtype=np.float32)
    read_value = ops.read_value(param, "MyVar", Type.f32, shape)
    model = Model(ops.relu(read_value), [param])

    ref_shape = model.input().partial_shape
    ref_shape[0] = 3
    model.reshape(ref_shape, {"MyVar": ref_shape})
    core = Core()
    compiled_model = core.compile_model(model, device)
    assert compiled_model.input().partial_shape == ref_shape


def test_reshape_with_python_types():
    model = generate_add_model()

    def check_shape(new_shape):
        for model_input in model.inputs:
            assert model_input.partial_shape == new_shape

    shape1 = [1, 4]
    new_shapes = {input: shape1 for input in model.inputs}
    model.reshape(new_shapes)
    check_shape(PartialShape(shape1))

    shape2 = [1, 6]
    new_shapes = {input.any_name: shape2 for input in model.inputs}
    model.reshape(new_shapes)
    check_shape(PartialShape(shape2))

    shape3 = [1, 8]
    new_shapes = {i: shape3 for i, _ in enumerate(model.inputs)}
    model.reshape(new_shapes)
    check_shape(PartialShape(shape3))

    shape4 = [1, -1]
    new_shapes = {input: shape4 for input in model.inputs}
    model.reshape(new_shapes)
    check_shape(PartialShape([Dimension(1), Dimension(-1)]))

    shape5 = [1, (1, 10)]
    new_shapes = {input: shape5 for input in model.inputs}
    model.reshape(new_shapes)
    check_shape(PartialShape([Dimension(1), Dimension(1, 10)]))

    shape6 = [Dimension(3), Dimension(3, 10)]
    new_shapes = {input: shape6 for input in model.inputs}
    model.reshape(new_shapes)
    check_shape(PartialShape(shape6))

    shape7 = "[1..10, ?]"
    new_shapes = {input: shape7 for input in model.inputs}
    model.reshape(new_shapes)
    check_shape(PartialShape(shape7))

    # reshape mixed keys
    shape8 = [(1, 20), -1]
    new_shapes = {"data1": shape8, 1: shape8}
    model.reshape(new_shapes)
    check_shape(PartialShape([Dimension(1, 20), Dimension(-1)]))

    # reshape with one input
    param = ops.parameter([1, 3, 28, 28])
    model = Model(ops.relu(param), [param])

    shape9 = [-1, 3, (28, 56), (28, 56)]
    model.reshape(shape9)
    check_shape(PartialShape([Dimension(-1), Dimension(3), Dimension(28, 56), Dimension(28, 56)]))

    shape10 = "[?,3,..224,..224]"
    model.reshape(shape10)
    check_shape(PartialShape([Dimension(-1), Dimension(3), Dimension(-1, 224), Dimension(-1, 224)]))

    # check exceptions
    shape10 = [1, 1, 1, 1]
    with pytest.raises(TypeError) as e:
        model.reshape({model.input().node: shape10})
    assert (
        "Incorrect key type <class 'openvino._pyopenvino.op.Parameter'> to reshape a model, "
        "expected keys as openvino.runtime.Output, int or str." in str(e.value)
    )

    with pytest.raises(TypeError) as e:
        model.reshape({0: range(1, 9)})
    assert (
        "Incorrect value type <class 'range'> to reshape a model, "
        "expected values as openvino.runtime.PartialShape, str, list or tuple."
        in str(e.value)
    )


def test_reshape_with_python_types_for_variable():
    var_id = "ID1"
    model = make_add_with_variable_model([1, 2, 5], var_id)

    def check_shape(new_shape):
        for model_input in model.inputs:
            assert model_input.partial_shape == new_shape

    shape1 = [1, 4]
    new_shapes = {input: PartialShape(shape1) for input in model.inputs}
    model.reshape(new_shapes, {var_id: shape1})
    check_shape(PartialShape(shape1))

    shape2 = [1, 6]
    new_shapes = {input.any_name: PartialShape(shape2) for input in model.inputs}
    model.reshape(new_shapes, {var_id: shape2})
    check_shape(PartialShape(shape2))

    shape3 = [1, 8]
    new_shapes = {i: PartialShape(shape3) for i, _ in enumerate(model.inputs)}
    model.reshape(new_shapes, {var_id: shape3})
    check_shape(PartialShape(shape3))

    shape4 = [1, -1]
    new_shapes = {input: PartialShape(shape4) for input in model.inputs}
    model.reshape(new_shapes, {var_id: shape4})
    check_shape(PartialShape([Dimension(1), Dimension(-1)]))

    shape5 = [1, (1, 10)]
    new_shapes = {input: PartialShape(shape5) for input in model.inputs}
    model.reshape(new_shapes, {var_id: shape5})
    check_shape(PartialShape([Dimension(1), Dimension(1, 10)]))

    shape6 = [Dimension(3), Dimension(3, 10)]
    new_shapes = {input: PartialShape(shape6) for input in model.inputs}
    model.reshape(new_shapes, {var_id: shape6})
    check_shape(PartialShape(shape6))

    shape7 = "[1..10, ?]"
    new_shapes = {input: PartialShape(shape7) for input in model.inputs}
    model.reshape(new_shapes, {var_id: shape7})
    check_shape(PartialShape(shape7))

    # reshape mixed keys
    shape8 = [(1, 20), -1]
    new_shapes = {"data1": PartialShape(shape8), 1: shape8}
    model.reshape(new_shapes, {var_id: shape8})
    check_shape(PartialShape([Dimension(1, 20), Dimension(-1)]))

    # reshape with one input
    param = ops.parameter([1, 3, 28, 28])
    read_value = ops.read_value(param, var_id, Type.f32, param.get_partial_shape())
    model = Model(ops.relu(read_value), [param])

    shape9 = [-1, 3, (28, 56), (28, 56)]
    model.reshape(shape9, {var_id: shape9})
    check_shape(PartialShape([Dimension(-1), Dimension(3), Dimension(28, 56), Dimension(28, 56)]))

    shape10 = "[?,3,..224,..224]"
    model.reshape(shape10, {var_id: shape10})
    check_shape(PartialShape([Dimension(-1), Dimension(3), Dimension(-1, 224), Dimension(-1, 224)]))

    # check exceptions
    shape10 = [1, 1, 1, 1]
    with pytest.raises(TypeError) as e:
        model.reshape({0: shape10}, {0: shape10})
    assert (
        "Incorrect key type <class 'int'> to reshape a model, expected values as str." in str(e.value)
    )

    with pytest.raises(TypeError) as e:
        model.reshape({0: shape10}, {var_id: range(1, 9)})
    assert (
        "Incorrect value type <class 'range'> to reshape a model, "
        "expected values as openvino.runtime.PartialShape, str, list or tuple."
        in str(e.value)
    )


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_serialize_rt_info(request, tmp_path):
    version = "TestVersion"
    config = "TestConfig"
    framework_batch = "1"

    def check_rt_info(model):
        assert model.get_rt_info("MO_version") == version
        assert model.get_rt_info(["Runtime_version"]) == version
        assert model.get_rt_info(["optimization", "config"]) == config
        assert model.get_rt_info(["framework", "batch"]) == framework_batch

        assert model.has_rt_info(["test"]) is False
        assert model.has_rt_info("optimization") is True
        assert model.has_rt_info(["optimization", "test"]) is False
        with pytest.raises(RuntimeError):
            assert model.get_rt_info(["test"])

        with pytest.raises(RuntimeError):
            assert model.get_rt_info(["optimization", "test"])

    core = Core()
    xml_path, bin_path = create_filename_for_test(request.node.name, tmp_path)
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    assert "relu_t1" in relu1.get_output_tensor(0).names
    relu2 = ops.relu(relu1, name="relu2")
    model = Model(relu2, [param], "TestModel")

    assert model is not None

    assert model.has_rt_info("MO_version") is False
    model.set_rt_info(version, "MO_version")
    assert model.has_rt_info("MO_version") is True

    assert model.has_rt_info(["Runtime_version"]) is False
    model.set_rt_info(version, ["Runtime_version"])
    assert model.has_rt_info(["Runtime_version"]) is True

    assert model.has_rt_info(["optimization"]) is False
    assert model.has_rt_info(["optimization", "config"]) is False
    model.set_rt_info(config, ["optimization", "config"])
    assert model.has_rt_info(["optimization"]) is True
    assert model.has_rt_info(["optimization", "config"]) is True

    assert model.has_rt_info(["framework"]) is False
    assert model.has_rt_info(["framework", "batch"]) is False
    model.set_rt_info(framework_batch, ["framework", "batch"])
    assert model.has_rt_info(["framework"]) is True
    assert model.has_rt_info(["framework", "batch"]) is True

    check_rt_info(model)

    serialize(model, xml_path, bin_path)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    check_rt_info(res_model)

    os.remove(xml_path)
    os.remove(bin_path)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_serialize_complex_rt_info(request, tmp_path):
    def check_rt_info(model):
        assert model.has_rt_info(["config", "type_of_model"]) is True
        assert model.has_rt_info(["config", "converter_type"]) is True
        assert model.has_rt_info(["config", "model_parameters", "threshold"]) is True
        assert model.has_rt_info(["config", "model_parameters", "min"]) is True
        assert model.has_rt_info(["config", "model_parameters", "max"]) is True
        assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "type"]) is True
        assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "directed"]) is True
        assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "float_empty"]) is True
        assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "nodes"]) is True
        assert model.has_rt_info(["config", "model_parameters", "labels", "label_groups", "ids"]) is True
        assert model.has_rt_info(["config", "model_parameters", "mean_values"]) is True

        assert model.get_rt_info(["config", "type_of_model"]).astype(str) == "classification"
        assert model.get_rt_info(["config", "converter_type"]).astype(str) == "classification"
        assert math.isclose(model.get_rt_info(["config", "model_parameters", "threshold"]).astype(float), 13.23, rel_tol=0.0001)
        assert math.isclose(model.get_rt_info(["config", "model_parameters", "min"]).astype(float), -3.24543, rel_tol=0.0001)
        assert math.isclose(model.get_rt_info(["config", "model_parameters", "max"]).astype(float), 3.234223, rel_tol=0.0001)
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "type"]).astype(str) == "tree"
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "directed"]).astype(bool) is True

        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "float_empty"]).aslist() == []
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "nodes"]).aslist() == []
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_groups", "ids"]).aslist(str) == ["sasd", "fdfdfsdf"]
        assert model.get_rt_info(["config", "model_parameters", "mean_values"]).aslist(float) == [22.3, 33.11, 44.0]

        rt_info = model.get_rt_info()
        assert isinstance(rt_info["config"], dict)

        for key, value in rt_info.items():
            if key == "config":
                for config_value in value:
                    assert config_value in ["type_of_model", "converter_type", "model_parameters"]

        for rt_info_val in model.get_rt_info(["config", "model_parameters", "labels", "label_tree"]).astype(dict):
            assert rt_info_val in ["float_empty", "nodes", "type", "directed"]

    core = Core()
    xml_path, bin_path = create_filename_for_test(request.node.name, tmp_path)
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    assert "relu_t1" in relu1.get_output_tensor(0).names
    relu2 = ops.relu(relu1, name="relu2")
    model = Model(relu2, [param], "TestModel")

    assert model is not None

    model.set_rt_info("classification", ["config", "type_of_model"])
    model.set_rt_info("classification", ["config", "converter_type"])
    model.set_rt_info(13.23, ["config", "model_parameters", "threshold"])
    model.set_rt_info(-3.24543, ["config", "model_parameters", "min"])
    model.set_rt_info(3.234223, ["config", "model_parameters", "max"])
    model.set_rt_info("tree", ["config", "model_parameters", "labels", "label_tree", "type"])
    model.set_rt_info(True, ["config", "model_parameters", "labels", "label_tree", "directed"])
    model.set_rt_info([], ["config", "model_parameters", "labels", "label_tree", "float_empty"])
    model.set_rt_info([], ["config", "model_parameters", "labels", "label_tree", "nodes"])
    model.set_rt_info(["sasd", "fdfdfsdf"], ["config", "model_parameters", "labels", "label_groups", "ids"])
    model.set_rt_info([22.3, 33.11, 44.0], ["config", "model_parameters", "mean_values"])

    check_rt_info(model)

    serialize(model, xml_path, bin_path)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    check_rt_info(res_model)

    os.remove(xml_path)
    os.remove(bin_path)


def test_model_add_remove_result_parameter_sink():
    param = ops.parameter(PartialShape([1]), dtype=np.float32, name="param")
    relu1 = ops.relu(param, name="relu1")
    relu2 = ops.relu(relu1, name="relu2")
    result = ops.result(relu2, "res")
    model = Model([result], [param], "TestModel")

    result2 = ops.result(relu2, "res2")
    model.add_results([result2])

    results = model.get_results()
    assert len(results) == 2
    assert results[0].get_output_element_type(0) == Type.f32
    assert results[0].get_output_partial_shape(0) == PartialShape([1])

    model.remove_result(result)
    assert len(model.results) == 1

    param1 = ops.parameter(PartialShape([1]), name="param1")
    model.add_parameters([param1])

    params = model.parameters
    assert (params[0].get_partial_shape()) == PartialShape([1])
    assert len(params) == 2

    model.remove_parameter(param)
    assert len(model.parameters) == 1

    assign = ops.assign()
    model.add_sinks([assign])

    assign_nodes = model.sinks
    assert ["Assign"] == [sink.get_type_name() for sink in assign_nodes]
    model.remove_sink(assign)
    assert len(model.sinks) == 0


def test_model_get_raw_address():
    model = generate_add_model()
    model_with_same_addr = model
    model_different = generate_add_model()

    assert model._get_raw_address() == model_with_same_addr._get_raw_address()
    assert model._get_raw_address() != model_different._get_raw_address()


def test_model_add_remove_variable():
    model = generate_model_with_memory(input_shape=Shape([2, 1]), data_type=Type.f32)

    var_info = VariableInfo()
    var_info.data_shape = PartialShape([2, 1])
    var_info.data_type = Type.f32
    var_info.variable_id = "v1"
    variable_1 = Variable(var_info)

    assert len(model.get_variables()) == 1
    model.add_variables([variable_1])
    assert len(model.get_variables()) == 2
    variable_by_id = model.get_variable_by_id("var_id_667")
    assert variable_by_id.info.variable_id == "var_id_667"
    model.remove_variable(variable_1)
    assert len(model.get_variables()) == 1


def test_save_model_with_none():
    with pytest.raises(AttributeError) as e:
        save_model(model=None, output_model="model.xml")
    assert "'model' argument is required and cannot be None." in str(e.value)


def test_copy_failed():
    model = generate_add_model()
    with pytest.raises(TypeError) as e:
        copy(model)
    assert "Cannot copy 'openvino.runtime.Model. Please, use deepcopy instead." in str(e.value)
