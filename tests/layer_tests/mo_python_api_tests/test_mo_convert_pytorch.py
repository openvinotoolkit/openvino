# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime as ov
import pytest
import torch
from openvino.runtime import PartialShape, Dimension, Model

from common.mo_convert_test_class import CommonMOConvertTest


def create_pytorch_nn_module_case1():
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = [sample_input1, sample_input2]

    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': [PartialShape([-1, 3, -1, -1]), PartialShape([-1, 3, -1, -1])],
                                       'sample_input': sample_input}


def create_pytorch_nn_module_case2():
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = [sample_input1, sample_input2]

    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': ["[?,3,?,?]", PartialShape([-1, 3, -1, -1])],
                                       'sample_input': sample_input, 'onnx_opset_version': 11}


def create_pytorch_nn_module_case3():
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = [sample_input1, sample_input2]

    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': "[?,3,?,?],[?,3,?,?]", 'sample_input': sample_input}


def create_pytorch_jit_script_module():
    import torch
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y):
            logits = self.linear_relu_stack(x + y)
            return logits

    net = NeuralNetwork()
    scripted_model = torch.jit.script(net)

    shape = PartialShape([1, 3, 5, 5])
    param1 = ov.opset8.parameter(shape, name="x.1", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="y.1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"onnx::Relu_2"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"result"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"logits"})

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return scripted_model, ref_model, {'input_shape': [PartialShape([1, 3, 5, 5]), PartialShape([1, 3, 5, 5])]}


def create_pytorch_jit_script_function():
    import torch

    @torch.jit.script
    def scripted_fn(x: torch.Tensor, y: torch.Tensor):
        return torch.sigmoid(torch.relu(x + y))

    inp_shape = PartialShape([Dimension(1, -1), Dimension(-1, 5), 10])

    shape = PartialShape([-1, -1, 10])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"onnx::Relu_2"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return scripted_fn, ref_model, {'input_shape': [inp_shape, inp_shape]}


class TestMoConvertPyTorch(CommonMOConvertTest):
    test_data = [
        create_pytorch_nn_module_case1,
        create_pytorch_nn_module_case2,
        create_pytorch_nn_module_case3,
        create_pytorch_jit_script_module,
        create_pytorch_jit_script_function,
    ]

    @pytest.mark.parametrize("create_model", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_import_from_memory(self, create_model, ie_device, precision, ir_version,
                                   temp_dir, use_new_frontend, use_old_api):
        fw_model, graph_ref, mo_params = create_model()

        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params, graph_ref)
