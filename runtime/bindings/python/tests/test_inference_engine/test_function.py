# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset8 as ops

from openvino.runtime import Function, Tensor
from openvino.runtime.descriptor import Tensor as DescriptorTensor

from openvino.runtime.impl import PartialShape, Shape


def test_function_add_outputs_tensor_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    assert "relu_t1" in relu1.get_output_tensor(0).names
    relu2 = ops.relu(relu1, name="relu2")
    function = Function(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    function.add_outputs("relu_t1")
    assert len(function.get_results()) == 2
    assert isinstance(function.outputs[1].get_tensor(), DescriptorTensor)
    assert "relu_t1" in function.outputs[1].get_tensor().names


def test_function_add_outputs_op_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Function(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    function.add_outputs(("relu1", 0))
    assert len(function.get_results()) == 2


def test_function_add_output_port():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Function(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    function.add_outputs(relu1.output(0))
    assert len(function.get_results()) == 2


def test_function_add_output_incorrect_tensor_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Function(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(RuntimeError) as e:
        function.add_outputs("relu_t")
    assert "Tensor name relu_t was not found." in str(e.value)


def test_function_add_output_incorrect_idx():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Function(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(RuntimeError) as e:
        function.add_outputs(("relu1", 10))
    assert "Cannot add output to port 10 operation relu1 has only 1 outputs." in str(
        e.value
    )


def test_function_add_output_incorrect_name():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Function(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(RuntimeError) as e:
        function.add_outputs(("relu_1", 0))
    assert "Port 0 for operation with name relu_1 was not found." in str(e.value)


def test_add_outputs_several_tensors():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    relu2.get_output_tensor(0).set_names({"relu_t2"})
    relu3 = ops.relu(relu2, name="relu3")
    function = Function(relu3, [param], "TestFunction")
    assert len(function.get_results()) == 1
    function.add_outputs(["relu_t1", "relu_t2"])
    assert len(function.get_results()) == 3


def test_add_outputs_several_ports():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    relu2.get_output_tensor(0).set_names({"relu_t2"})
    relu3 = ops.relu(relu2, name="relu3")
    function = Function(relu3, [param], "TestFunction")
    assert len(function.get_results()) == 1
    function.add_outputs([("relu1", 0), ("relu2", 0)])
    assert len(function.get_results()) == 3


def test_add_outputs_incorrect_value():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    relu2 = ops.relu(relu1, name="relu2")
    function = Function(relu2, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(TypeError) as e:
        function.add_outputs(0)
    assert "Incorrect type of a value to add as output." in str(e.value)


def test_add_outputs_incorrect_outputs_list():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu1 = ops.relu(param, name="relu1")
    relu1.get_output_tensor(0).set_names({"relu_t1"})
    function = Function(relu1, [param], "TestFunction")
    assert len(function.get_results()) == 1
    with pytest.raises(TypeError) as e:
        function.add_outputs([0, 0])
    assert "Incorrect type of a value to add as output at index 0" in str(e.value)


def test_validate_nodes_and_infer_types():
    param1 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)
    func = Function(add, [param1, param2], "TestFunction")

    invalid_shape = Shape([3, 7])
    param3 = ops.parameter(invalid_shape, dtype=np.float32, name="data3")
    func.replace_parameter(0, param3)

    with pytest.raises(RuntimeError) as e:
        func.validate_nodes_and_infer_types()
    assert "Argument shapes are inconsistent" in str(e.value)


def test_get_result_index():
    input_shape = PartialShape([1])
    param = ops.parameter(input_shape, dtype=np.float32, name="data")
    relu = ops.relu(param, name="relu")
    function = Function(relu, [param], "TestFunction")
    assert len(function.outputs) == 1
    assert function.get_result_index(function.outputs[0]) == 0


def test_get_result_index_invalid():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data1")
    relu1 = ops.relu(param1, name="relu1")
    function = Function(relu1, [param1], "TestFunction")

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
    function = Function(relu, [param], "TestFunction")
    assert function.get_parameter_index(param) == 0


def test_parameter_index_invalid():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data1")
    relu = ops.relu(param1, name="relu")
    function = Function(relu, [param1], "TestFunction")
    shape2 = PartialShape([2])
    param2 = ops.parameter(shape2, dtype=np.float32, name="data2")
    assert function.get_parameter_index(param2) == -1


def test_replace_parameter():
    shape1 = PartialShape([1])
    param1 = ops.parameter(shape1, dtype=np.float32, name="data")
    shape2 = PartialShape([2])
    param2 = ops.parameter(shape2, dtype=np.float32, name="data")
    relu = ops.relu(param1, name="relu")

    function = Function(relu, [param1], "TestFunction")
    param_index = function.get_parameter_index(param1)
    function.replace_parameter(param_index, param2)
    assert function.get_parameter_index(param2) == param_index
    assert function.get_parameter_index(param1) == -1


def test_evaluate():
    param1 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)
    func = Function(add, [param1, param2], "TestFunction")

    input1 = np.array([2, 1], dtype=np.float32).reshape(2, 1)
    input2 = np.array([3, 7], dtype=np.float32).reshape(2, 1)
    out_tensor = Tensor("float32", Shape([2, 1]))

    assert func.evaluate([out_tensor], [Tensor(input1), Tensor(input2)])
    assert np.allclose(out_tensor.data, np.array([5, 8]).reshape(2, 1))


def test_evaluate_invalid_input_shape():
    param1 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data1")
    param2 = ops.parameter(Shape([2, 1]), dtype=np.float32, name="data2")
    add = ops.add(param1, param2)
    func = Function(add, [param1, param2], "TestFunction")

    with pytest.raises(RuntimeError) as e:
        assert func.evaluate(
            [Tensor("float32", Shape([2, 1]))],
            [Tensor("float32", Shape([3, 1])), Tensor("float32", Shape([3, 1]))],
        )
    assert "must be compatible with the partial shape: {2,1}" in str(e.value)
