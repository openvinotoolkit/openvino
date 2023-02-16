# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import pytest

import openvino.runtime.opset8 as ops
from openvino.runtime import (
    Core,
    Model,
    Tensor,
    Output,
    Dimension,
    Layout,
    Type,
    PartialShape,
    Shape,
    set_batch,
    get_batch,
    serialize,
)

from tests.test_utils.test_utils import generate_add_model, create_filename_for_test


def test_test_descriptor_tensor():
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


def test_function_add_outputs_tensor_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    assert "relu_t1" in relu1.get_output_tensor(0).names
    relu2 = ops.relu(relu1, name="relu2")
    function = Model(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    new_outs = function.add_outputs("relu_t1")
    assert len(function.get_results()) == 2
    assert "relu_t1" in function.outputs[1].get_tensor().names
    assert len(new_outs) == 1
    assert new_outs[0].get_node() == function.outputs[1].get_node()
    assert new_outs[0].get_index() == function.outputs[1].get_index()


def test_function_add_outputs_op_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Model(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    new_outs = function.add_outputs(("relu1", 0))
    assert len(function.get_results()) == 2
    assert len(new_outs) == 1
    assert new_outs[0].get_node() == function.outputs[1].get_node()
    assert new_outs[0].get_index() == function.outputs[1].get_index()


def test_function_add_output_port():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Model(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    new_outs = function.add_outputs(relu1.output(0))
    assert len(function.get_results()) == 2
    assert len(new_outs) == 1
    assert new_outs[0].get_node() == function.outputs[1].get_node()
    assert new_outs[0].get_index() == function.outputs[1].get_index()


def test_function_add_output_incorrect_tensor_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Model(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(RuntimeError) as e:
        function.add_outputs("relu_t")
    # Verify that absent output name is present in error message
    assert "relu_t" in str(e.value)


def test_function_add_output_incorrect_idx():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Model(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(RuntimeError) as e:
        function.add_outputs(("relu1", 1234))
    # Verify that op name and port number are present in error message
    assert "relu1" in str(e.value)
    assert "1234" in str(e.value)


def test_function_add_output_incorrect_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Model(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(RuntimeError) as e:
        function.add_outputs(("relu_1", 0))
    # Verify that absent op name is present in error message
    assert "relu_1" in str(e.value)


def test_add_outputs_several_tensors():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    relu2.get_output_tensor(0).set_names({"relu_t2"})
    relu3 = ops.relu(relu2, name="relu3")
    function = Model(relu3, [param], "TestFunction")
    assert len(function.get_results()) == 1
    new_outs = function.add_outputs(["relu_t1", "relu_t2"])
    assert len(function.get_results()) == 3
    assert len(new_outs) == 2
    assert new_outs[0].get_node() == function.outputs[1].get_node()
    assert new_outs[0].get_index() == function.outputs[1].get_index()
    assert new_outs[1].get_node() == function.outputs[2].get_node()
    assert new_outs[1].get_index() == function.outputs[2].get_index()


def test_add_outputs_several_ports():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    relu2.get_output_tensor(0).set_names({"relu_t2"})
    relu3 = ops.relu(relu2, name="relu3")
    function = Model(relu3, [param], "TestFunction")
    assert len(function.get_results()) == 1
    new_outs = function.add_outputs([("relu1", 0), ("relu2", 0)])
    assert len(function.get_results()) == 3
    assert len(new_outs) == 2
    assert new_outs[0].get_node() == function.outputs[1].get_node()
    assert new_outs[0].get_index() == function.outputs[1].get_index()
    assert new_outs[1].get_node() == function.outputs[2].get_node()
    assert new_outs[1].get_index() == function.outputs[2].get_index()


def test_add_outputs_incorrect_value():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Model(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(TypeError) as e:
        function.add_outputs(0)
    assert "Incorrect type of a value to add as output." in str(e.value)


def test_add_outputs_incorrect_outputs_list():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    function = Model(relu1, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(TypeError) as e:
        function.add_outputs([0, 0])
    assert "Incorrect type of a value to add as output at index 0" in str(e.value)


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
    function = Model(relu, [param], "TestFunction")
    assert len(function.outputs) == 1
    assert function.get_result_index(function.outputs[0]) == 0


def test_get_result_index_invalid():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data1")
    relu1 = ops.relu(param1, name="relu1")
    function = Model(relu1, [param1], "TestFunction")

    shape2 = PartialShape([2])
    param2 = ops.parameter(shape2, dtype=np.float32, name="data2")
    relu2 = ops.relu(param2, name="relu2")
    invalid_output = relu2.outputs()[0]
    assert len(function.outputs) == 1
    assert function.get_result_index(invalid_output) == -1


def test_parameter_index():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu = ops.relu(param, name="relu")
    function = Model(relu, [param], "TestFunction")
    assert function.get_parameter_index(param) == 0


def test_parameter_index_invalid():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data1")
    relu = ops.relu(param1, name="relu")
    function = Model(relu, [param1], "TestFunction")
    shape2 = PartialShape([2])
    param2 = ops.parameter(shape2, dtype=np.float32, name="data2")
    assert function.get_parameter_index(param2) == -1


def test_replace_parameter():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data")
    shape2 = PartialShape([2])
    param2 = ops.parameter(shape2, dtype=np.float32, name="data")
    relu = ops.relu(param1, name="relu")

    function = Model(relu, [param1], "TestFunction")
    param_index = function.get_parameter_index(param1)
    function.replace_parameter(param_index, param2)
    assert function.get_parameter_index(param2) == param_index
    assert function.get_parameter_index(param1) == -1


def test_evaluate():
    model = generate_add_model()
    input1 = np.array([2, 1], dtype=np.float32).reshape(2, 1)
    input2 = np.array([3, 7], dtype=np.float32).reshape(2, 1)
    out_tensor = Tensor("float32", Shape([2, 1]))

    assert model.evaluate([out_tensor], [Tensor(input1), Tensor(input2)])
    assert np.allclose(out_tensor.data, np.array([5, 8]).reshape(2, 1))


def test_evaluate_invalid_input_shape():
    model = generate_add_model()
    with pytest.raises(RuntimeError) as e:
        assert model.evaluate(
            [Tensor("float32", Shape([2, 1]))],
            [Tensor("float32", Shape([3, 1])), Tensor("float32", Shape([3, 1]))],
        )
    assert "must be compatible with the partial shape: [2,1]" in str(e.value)


def test_get_batch():
    model = generate_add_model()
    param = model.get_parameters()[0]
    param.set_layout(Layout("NC"))
    assert get_batch(model) == 2


def test_get_batch_chwn():
    param1 = ops.parameter(Shape([3, 1, 3, 4]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([3, 1, 3, 4]), dtype=np.float32, name="data2")
    param3 = ops.parameter(Shape([3, 1, 3, 4]), dtype=np.float32, name="data3")
    add = ops.add(param1, param2)
    add2 = ops.add(add, param3)
    model = Model(add2, [param1, param2, param3], "TestFunction")
    param = model.get_parameters()[0]
    param.set_layout(Layout("CHWN"))
    assert get_batch(model) == 4


def test_set_batch_dimension():
    model = generate_add_model()
    model_param1 = model.get_parameters()[0]
    model_param2 = model.get_parameters()[1]
    # check batch == 2
    model_param1.set_layout(Layout("NC"))
    assert get_batch(model) == 2
    # set batch to 1
    set_batch(model, Dimension(1))
    assert get_batch(model) == 1
    # check if shape of param 1 has changed
    assert model_param1.get_output_shape(0) == PartialShape([1, 1])
    # check if shape of param 2 has not changed
    assert model_param2.get_output_shape(0) == PartialShape([2, 1])


def test_set_batch_int():
    model = generate_add_model()
    model_param1 = model.get_parameters()[0]
    model_param2 = model.get_parameters()[1]
    # check batch == 2
    model_param1.set_layout(Layout("NC"))
    assert get_batch(model) == 2
    # set batch to 1
    set_batch(model, 1)
    assert get_batch(model) == 1
    # check if shape of param 1 has changed
    assert model_param1.get_output_shape(0) == PartialShape([1, 1])
    # check if shape of param 2 has not changed
    assert model_param2.get_output_shape(0) == PartialShape([2, 1])


def test_set_batch_default_batch_size():
    model = generate_add_model()
    model_param1 = model.get_parameters()[0]
    model_param1.set_layout(Layout("NC"))
    set_batch(model)
    assert model.is_dynamic()


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


def test_reshape_with_python_types(device):
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
    model = Model(relu2, [param], "TestFunction")

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
    def check_rt_info(model, serialized):
        if serialized:
            threshold = "13.23"
            min_val = "-3.24543"
            max_val = "3.23422"
            directed = "YES"
            empty = ""
            ids = "sasd fdfdfsdf"
            mean = "22.3 33.11 44"
        else:
            threshold = 13.23
            min_val = -3.24543
            max_val = 3.234223
            directed = True
            empty = []
            ids = ["sasd", "fdfdfsdf"]
            mean = [22.3, 33.11, 44.0]
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

        assert model.get_rt_info(["config", "type_of_model"]) == "classification"
        assert model.get_rt_info(["config", "converter_type"]) == "classification"
        assert model.get_rt_info(["config", "model_parameters", "threshold"]) == threshold
        assert model.get_rt_info(["config", "model_parameters", "min"]) == min_val
        assert model.get_rt_info(["config", "model_parameters", "max"]) == max_val
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "type"]) == "tree"
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "directed"]) == directed
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "float_empty"]) == empty
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "nodes"]) == empty
        assert model.get_rt_info(["config", "model_parameters", "labels", "label_groups", "ids"]) == ids
        assert model.get_rt_info(["config", "model_parameters", "mean_values"]) == mean

        rt_info = model.get_rt_info()
        assert rt_info["config"] == {"converter_type": "classification",
                                     "model_parameters": {"labels": {"label_groups": {"ids": ids},
                                                          "label_tree": {"directed": directed, "float_empty": empty,
                                                                         "nodes": empty, "type": "tree"}},
                                                          "max": max_val, "mean_values": mean,
                                                          "min": min_val, "threshold": threshold},
                                     "type_of_model": "classification"}

        for key, value in rt_info.items():
            if key == "config":
                for config_value in value:
                    assert config_value in ["type_of_model", "converter_type", "model_parameters"]

        for rt_info_val in model.get_rt_info(["config", "model_parameters", "labels", "label_tree"]):
            assert rt_info_val in ["float_empty", "nodes", "type", "directed"]

    core = Core()
    xml_path, bin_path = create_filename_for_test(request.node.name, tmp_path)
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    assert "relu_t1" in relu1.get_output_tensor(0).names
    relu2 = ops.relu(relu1, name="relu2")
    model = Model(relu2, [param], "TestFunction")

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

    check_rt_info(model, False)

    serialize(model, xml_path, bin_path)

    res_model = core.read_model(model=xml_path, weights=bin_path)

    check_rt_info(res_model, True)

    os.remove(xml_path)
    os.remove(bin_path)
