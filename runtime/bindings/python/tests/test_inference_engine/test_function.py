# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.opset8 as ops

from openvino import Function
from openvino.descriptor import Tensor
from openvino.impl import PartialShape


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
    assert isinstance(function.outputs[1].get_tensor(), Tensor)
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
    assert "Cannot add output to port 10 operation relu1 has only 1 outputs." in str(e.value)


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
