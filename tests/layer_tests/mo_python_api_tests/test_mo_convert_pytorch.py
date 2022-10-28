# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import pathlib

import numpy
import numpy as np
import os
import openvino.runtime as ov
import pytest
import torch
from openvino.runtime import PartialShape, Dimension, Model

from common.mo_convert_test_class import CommonMOConvertTest


def create_pytorch_nn_module_case1(tmp_dir):
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
                                       'example_inputs': sample_input}


def create_pytorch_nn_module_case2(tmp_dir):
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
                                       'example_inputs': sample_input, 'onnx_opset_version': 11}


def create_pytorch_nn_module_case3(tmp_dir):
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

    return NeuralNetwork(), function, {'input_shape': "[?,3,?,?],[?,3,?,?]", 'example_inputs': sample_input}


def create_pytorch_nn_module_case4(tmp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    sample_input = torch.zeros(1, 3, 10, 10)

    shape = PartialShape([1, 3, 10, 10])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.float32)
    param1.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(param1)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_1"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"2"})

    parameter_list = [param1]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'example_inputs': sample_input}


def create_pytorch_nn_module_case5(tmp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    sample_input = torch.zeros(3, 3, 10, 10)

    shape = PartialShape([-1, 3, 10, 10])
    param1 = ov.opset8.parameter(shape, name="input", dtype=np.float32)
    param1.get_output_tensor(0).set_names({"input_0"})
    relu = ov.opset8.relu(param1)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_1"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"2"})

    parameter_list = [param1]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'example_inputs': sample_input,
                                       'input_shape': PartialShape([-1, 3, Dimension(2, -1), Dimension(-1, 10)])}


def create_pytorch_nn_module_case6(tmp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    shape = PartialShape([1, 3, 2, 10])
    param1 = ov.opset8.parameter(shape, name="input", dtype=np.float32)
    param1.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(param1)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_1"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"2"})

    parameter_list = [param1]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': PartialShape([1, 3, Dimension(2, -1), Dimension(-1, 10)])}


def create_pytorch_nn_module_torch_size(tmp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    shape = PartialShape([1, 3, 2, 10])
    param1 = ov.opset8.parameter(shape, name="input", dtype=np.float32)
    param1.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(param1)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_1"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"2"})

    parameter_list = [param1]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': torch.Size([1, 3, 2, 10])}


def create_pytorch_nn_module_sample_input_int32(tmp_dir):
    from torch import nn
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)

    shape = PartialShape([-1, 3, 10, 10])
    param1 = ov.opset8.parameter(shape, name="input", dtype=np.int32)
    param1.get_output_tensor(0).set_names({"input_0"})
    relu = ov.opset8.relu(param1)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_1"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"2"})

    parameter_list = [param1]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'example_inputs': sample_input,
                                       'input_shape': PartialShape([-1, 3, Dimension(2, -1), Dimension(-1, 10)])}


def create_pytorch_nn_module_sample_input_int32_two_inputs(tmp_dir):
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

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input = [sample_input1, sample_input2]

    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=np.int32)
    param2 = ov.opset8.parameter(shape, name="input_1", dtype=np.int32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"input"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_4", "onnx::Cast_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"5"})

    parameter_list = [param1, param2]
    function = Model([sigm], parameter_list, "test")

    return NeuralNetwork(), function, {'input_shape': ["[?,3,?,?]", PartialShape([-1, 3, -1, -1])],
                                       'example_inputs': sample_input, 'onnx_opset_version': 11}


def create_pytorch_nn_module_compare_convert_paths_case1(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)
    return NeuralNetwork(), ref_model, {'example_inputs': sample_input, 'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case2(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)
    return NeuralNetwork(), ref_model, {'example_inputs': sample_input,
                                        'input_shape': [1, 3, 10, 10],
                                        'onnx_opset_version': 16}

def create_pytorch_nn_module_compare_convert_paths_case3(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)
    return NeuralNetwork(), ref_model, {'input_shape': [1, 3, 10, 10],
                                        'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case4(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
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

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input = (sample_input1, sample_input2)

    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)

    return NeuralNetwork(), ref_model, {'example_inputs': sample_input, 'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case5(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
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

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input = tuple([sample_input1, sample_input2])

    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)

    return NeuralNetwork(), ref_model, {'example_inputs': sample_input,
                                        'input_shape': [torch.Size([1, 3, 10, 10]), PartialShape([1, 3, 10, 10])],
                                        'onnx_opset_version': 16}

def create_pytorch_nn_module_compare_convert_paths_case6(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
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

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
    sample_input = tuple([sample_input1, sample_input2])

    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)

    return NeuralNetwork(), ref_model, {'input_shape': [torch.Size([1, 3, 10, 10]), torch.Size([1, 3, 10, 10])],
                                        'onnx_opset_version': 16}


def create_pytorch_jit_script_module(tmp_dir):
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


def create_pytorch_jit_script_function(tmp_dir):
    import torch

    @torch.jit.script
    def scripted_fn(x: torch.Tensor, y: torch.Tensor):
        return torch.sigmoid(torch.relu(x + y))

    inp_shape = PartialShape([Dimension(1, -1), Dimension(-1, 5), 10])

    shape = PartialShape([1, 5, 10])
    param1 = ov.opset8.parameter(shape, name="x.1", dtype=np.float32)
    param2 = ov.opset8.parameter(shape, name="y.1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    add.get_output_tensor(0).set_names({"onnx::Relu_2"})
    relu = ov.opset8.relu(add)
    relu.get_output_tensor(0).set_names({"onnx::Sigmoid_3"})
    sigm = ov.opset8.sigmoid(relu)
    sigm.get_output_tensor(0).set_names({"4"})

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return scripted_fn, ref_model, {'input_shape': [inp_shape, inp_shape]}

def create_pytorch_nn_module_sample_input_numpy(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    example_inputs = np.array(torch.zeros(1, 3, 10, 10, dtype=torch.int32))
    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, torch.zeros(1, 3, 10, 10, dtype=torch.int32), onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)
    return NeuralNetwork(), ref_model, {'example_inputs': example_inputs,
                                        'input_shape': [1, 3, 10, 10],
                                        'onnx_opset_version': 16}


def create_pytorch_nn_module_sample_input_ov_host_tensor(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
    from openvino.runtime import Tensor
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    sample_input = Tensor(np.zeros([1, 3, 10, 10], dtype=np.int32))
    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, torch.zeros(1, 3, 10, 10, dtype=torch.int32), onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)
    return NeuralNetwork(), ref_model, {'example_inputs': sample_input,
                                        'input_shape': [1, 3, 10, 10],
                                        'onnx_opset_version': 16}



def create_pytorch_nn_module_sample_input_ov_host_tensor_two_inputs(tmp_dir):
    from torch import nn
    from openvino.tools.mo import convert_model
    from openvino.runtime import Tensor
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

    sample_input1 = Tensor(np.zeros([1, 3, 10, 10], dtype=np.int32))
    sample_input2 = Tensor(np.zeros([1, 3, 10, 10], dtype=np.int32))
    sample_input = sample_input1, sample_input2

    pt_model = NeuralNetwork()
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, tuple([torch.zeros(1, 3, 10, 10, dtype=torch.int32),
                                      torch.zeros(1, 3, 10, 10, dtype=torch.int32)]),
                      onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path)

    return NeuralNetwork(), ref_model, {'example_inputs': sample_input,
                                        'onnx_opset_version': 16}


class TestMoConvertPyTorch(CommonMOConvertTest):
    test_data = [
        create_pytorch_nn_module_case1,
        create_pytorch_nn_module_case2,
        create_pytorch_nn_module_case3,
        create_pytorch_nn_module_case4,
        create_pytorch_nn_module_case5,
        create_pytorch_nn_module_case6,
        create_pytorch_nn_module_torch_size,
        create_pytorch_nn_module_sample_input_int32,
        create_pytorch_nn_module_sample_input_int32_two_inputs,
        create_pytorch_nn_module_compare_convert_paths_case1,
        create_pytorch_nn_module_compare_convert_paths_case2,
        create_pytorch_nn_module_compare_convert_paths_case4,
        create_pytorch_nn_module_compare_convert_paths_case5,
        create_pytorch_nn_module_compare_convert_paths_case6,
        create_pytorch_nn_module_sample_input_numpy,
        create_pytorch_nn_module_sample_input_ov_host_tensor,
        create_pytorch_nn_module_sample_input_ov_host_tensor_two_inputs,
        create_pytorch_jit_script_module,
        create_pytorch_jit_script_function,
    ]

    @pytest.mark.parametrize("create_model", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_import_from_memory(self, create_model, ie_device, precision, ir_version,
                                   temp_dir, use_new_frontend, use_old_api):
        fw_model, graph_ref, mo_params = create_model(temp_dir)

        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params, graph_ref)
