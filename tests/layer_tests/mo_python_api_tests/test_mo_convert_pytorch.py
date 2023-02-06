# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy
import numpy as np
import openvino.runtime as ov
import pytest
import torch
from openvino.runtime import PartialShape, Dimension, Model

from common.mo_convert_test_class import CommonMOConvertTest


def make_pt_model_one_input():
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

    return NeuralNetwork()


def make_pt_model_two_inputs():
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

    return NeuralNetwork()


def make_ref_pt_model_one_input(shape, dtype=np.float32):
    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=dtype)
    relu = ov.opset8.relu(param1)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1]
    model = Model([sigm], parameter_list, "test")
    return model


def make_ref_pt_model_two_inputs(shape, dtype=np.float32):
    if len(shape) == 2:
        param1 = ov.opset8.parameter(PartialShape(shape[0]), name="input_0", dtype=dtype)
        param2 = ov.opset8.parameter(PartialShape(shape[1]), name="input_1", dtype=dtype)
    else:
        shape = PartialShape(shape)
        param1 = ov.opset8.parameter(shape, name="input_0", dtype=dtype)
        param2 = ov.opset8.parameter(shape, name="input_1", dtype=dtype)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model = Model([sigm], parameter_list, "test")
    return model


def create_pytorch_nn_module_case1(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([-1, 3, -1, -1])

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = sample_input1, sample_input2

    return pt_model, ref_model, {'input_shape': [PartialShape([-1, 3, -1, -1]), PartialShape([-1, 3, -1, -1])],
                                 'example_input': sample_input}


def create_pytorch_nn_module_case2(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([-1, 3, -1, -1])

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = sample_input1, sample_input2

    return pt_model, ref_model, {'input_shape': ["[?,3,?,?]", PartialShape([-1, 3, -1, -1])],
                                 'example_input': sample_input, 'onnx_opset_version': 11}


def create_pytorch_nn_module_case3(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([-1, 3, -1, -1])

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = tuple([sample_input1, sample_input2])

    return pt_model, ref_model, {'input_shape': "[?,3,?,?],[?,3,?,?]", 'example_input': sample_input}


def create_pytorch_nn_module_case4(tmp_dir):
    pt_model = make_pt_model_one_input()

    sample_input = torch.zeros(1, 3, 10, 10)

    ref_model = make_ref_pt_model_one_input([1, 3, 10, 10])

    return pt_model, ref_model, {'example_input': sample_input}


def create_pytorch_nn_module_case5(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shape = PartialShape([-1, 3, Dimension(2, -1), Dimension(-1, 10)])
    ref_model = make_ref_pt_model_one_input(inp_shape)

    sample_input = torch.zeros(3, 3, 10, 10)
    return pt_model, ref_model, {'example_input': sample_input,
                                 'input_shape': inp_shape}


def create_pytorch_nn_module_case6(tmp_dir):
    pt_model = make_pt_model_one_input()
    shape = PartialShape([1, 3, Dimension(2, -1), Dimension(-1, 10)])
    ref_model = make_ref_pt_model_one_input(shape)

    return pt_model, ref_model, {'input_shape': shape}


def create_pytorch_nn_module_torch_size(tmp_dir):
    pt_model = make_pt_model_one_input()
    ref_model = make_ref_pt_model_one_input([1, 3, 2, 10])

    return pt_model, ref_model, {'input_shape': torch.Size([1, 3, 2, 10])}


def create_pytorch_nn_module_sample_input_int32(tmp_dir):
    pt_model = make_pt_model_one_input()
    shape = PartialShape([-1, 3, Dimension(2, -1), Dimension(-1, 10)])

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)

    ref_model = make_ref_pt_model_one_input(shape, dtype=numpy.int32)

    return pt_model, ref_model, {'example_input': sample_input,
                                 'input_shape': shape}


def create_pytorch_nn_module_sample_input_int32_two_inputs(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    inp_shapes = ["[?,3,?,?]", PartialShape([-1, 3, -1, -1])]

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input = sample_input1, sample_input2
    ref_model = make_ref_pt_model_two_inputs([PartialShape([-1, 3, -1, -1]), inp_shapes[1]], dtype=np.int32)

    return pt_model, ref_model, {'input_shape': inp_shapes,
                                 'example_input': sample_input, 'onnx_opset_version': 11}


def create_pytorch_nn_module_compare_convert_paths_case1(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_one_input()

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'example_input': sample_input, 'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case2(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_one_input()

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'example_input': sample_input,
                                 'input_shape': [1, 3, 10, 10],
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case3(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_one_input()

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'input_shape': [1, 3, 10, 10],
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case4(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_two_inputs()

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input = (sample_input1, sample_input2)

    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)

    return pt_model, ref_model, {'example_input': sample_input, 'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case5(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_two_inputs()

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input = tuple([sample_input1, sample_input2])

    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)

    return pt_model, ref_model, {'example_input': sample_input,
                                 'input_shape': [torch.Size([1, 3, 10, 10]), PartialShape([1, 3, 10, 10])],
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_compare_convert_paths_case6(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_two_inputs()

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.float32)
    sample_input = tuple([sample_input1, sample_input2])

    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, sample_input, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)

    return pt_model, ref_model, {'input_shape': [torch.Size([1, 3, 10, 10]), torch.Size([1, 3, 10, 10])],
                                 'onnx_opset_version': 16}


def create_pytorch_jit_script_module(tmp_dir):
    import torch

    net = make_pt_model_two_inputs()
    scripted_model = torch.jit.script(net)

    model_ref = make_ref_pt_model_two_inputs([1, 3, 5, 5])
    return scripted_model, model_ref, {'input_shape': [PartialShape([1, 3, 5, 5]), PartialShape([1, 3, 5, 5])]}


def create_pytorch_jit_script_function(tmp_dir):
    import torch

    @torch.jit.script
    def scripted_fn(x: torch.Tensor, y: torch.Tensor):
        return torch.sigmoid(torch.relu(x + y))

    inp_shape = PartialShape([Dimension(1, -1), Dimension(-1, 5), 10])
    ref_model = make_ref_pt_model_two_inputs(inp_shape)
    return scripted_fn, ref_model, {'input_shape': [inp_shape, inp_shape]}


def create_pytorch_nn_module_sample_input_numpy(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_one_input()

    example_inputs = np.array(torch.zeros(1, 3, 10, 10, dtype=torch.int32))
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, torch.zeros(1, 3, 10, 10, dtype=torch.int32), onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'example_input': example_inputs,
                                 'input_shape': [1, 3, 10, 10],
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_sample_input_dict(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_one_input()

    example_inputs = {"x": np.array(torch.zeros(1, 3, 10, 10, dtype=torch.int32))}
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, torch.zeros(1, 3, 10, 10, dtype=torch.int32), onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'example_input': example_inputs,
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_sample_input_dict_two_inputs(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_two_inputs()

    example_inputs = {"y": np.array(torch.zeros(1, 3, 10, 10, dtype=torch.int32)),
                      "x": np.array(torch.zeros(1, 3, 10, 10, dtype=torch.int32))}
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, {"y": torch.zeros(1, 3, 10, 10, dtype=torch.int32),
                                 "x": torch.zeros(1, 3, 10, 10, dtype=torch.int32)}, onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'example_input': example_inputs,
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_sample_list_of_tensors(tmp_dir):
    from openvino.tools.mo import convert_model
    pt_model = make_pt_model_one_input()

    example_inputs = [torch.zeros(3, 10, 10, dtype=torch.float32)]

    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, torch.unsqueeze(example_inputs[0], 0), onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'example_input': example_inputs,
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_sample_input_ov_host_tensor(tmp_dir):
    from openvino.tools.mo import convert_model
    from openvino.runtime import Tensor
    pt_model = make_pt_model_one_input()

    sample_input = Tensor(np.zeros([1, 3, 10, 10], dtype=np.int32))
    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, torch.zeros(1, 3, 10, 10, dtype=torch.int32), onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)
    return pt_model, ref_model, {'example_input': sample_input,
                                 'input_shape': [1, 3, 10, 10],
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_sample_input_ov_host_tensor_two_inputs(tmp_dir):
    from openvino.tools.mo import convert_model
    from openvino.runtime import Tensor
    pt_model = make_pt_model_two_inputs()

    sample_input1 = Tensor(np.zeros([1, 3, 10, 10], dtype=np.int32))
    sample_input2 = Tensor(np.zeros([1, 3, 10, 10], dtype=np.int32))
    sample_input = sample_input1, sample_input2

    onnx_model_path = os.path.join(tmp_dir, 'export.onnx')
    torch.onnx.export(pt_model, tuple([torch.zeros(1, 3, 10, 10, dtype=torch.int32),
                                       torch.zeros(1, 3, 10, 10, dtype=torch.int32)]),
                      onnx_model_path, opset_version=16)

    ref_model = convert_model(onnx_model_path, compress_to_fp16=False)

    return pt_model, ref_model, {'example_input': sample_input,
                                 'onnx_opset_version': 16}


def create_pytorch_nn_module_layout_list(tmp_dir):
    from openvino.runtime import Layout
    pt_model = make_pt_model_two_inputs()
    shape = [1, 3, 10, 10]

    shape = PartialShape(shape)
    ref_model = make_ref_pt_model_two_inputs(shape)
    ref_model.inputs[0].node.layout = Layout('nchw')
    ref_model.inputs[1].node.layout = Layout('nhwc')

    return pt_model, ref_model, {'input_shape': [shape, shape], 'layout': ['nchw', Layout('nhwc')],
                                 'onnx_opset_version': 11}


def create_pytorch_nn_module_layout_list_case2(tmp_dir):
    from openvino.runtime import Layout
    pt_model = make_pt_model_two_inputs()
    shape = [1, 3, 10, 10]

    shape = PartialShape(shape)
    ref_model = make_ref_pt_model_two_inputs(shape)
    ref_model.inputs[0].node.layout = Layout('nchw')
    ref_model.inputs[1].node.layout = Layout('nhwc')

    return pt_model, ref_model, {'input_shape': [shape, shape], 'layout': ('nchw', Layout('nhwc')),
                                 'onnx_opset_version': 11}


def create_pytorch_nn_module_mean_list(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    shape = [1, 10, 10, 3]

    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape)
    param2 = ov.opset8.parameter(shape)
    const1 = ov.opset8.constant([[[[0, 0, 0]]]], dtype=np.float32)
    const2 = ov.opset8.constant([[[[0, 0, 0]]]], dtype=np.float32)
    sub1 = ov.opset8.subtract(param1, const1)
    sub2 = ov.opset8.subtract(param2, const2)
    add = ov.opset8.add(sub1, sub2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {'input_shape': [shape, shape], 'mean_values': [[0, 0, 0], [0, 0, 0]],
                                 'onnx_opset_version': 11}


def create_pytorch_nn_module_scale_list(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    shape = [1, 10, 10, 3]

    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape)
    param2 = ov.opset8.parameter(shape)
    const1 = ov.opset8.constant([[[[1, 1, 1]]]], dtype=np.float32)
    const2 = ov.opset8.constant([[[[1, 1, 1]]]], dtype=np.float32)
    sub1 = ov.opset8.multiply(param1, const1)
    sub2 = ov.opset8.multiply(param2, const2)
    add = ov.opset8.add(sub1, sub2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {'input_shape': [shape, shape], 'scale_values': [[1, 1, 1], [1, 1, 1]],
                                 'onnx_opset_version': 11}


def create_pytorch_nn_module_shapes_list_static(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([1, 3, 20, 20])

    return pt_model, ref_model, {'input_shape': [[1, 3, 20, 20], [1, 3, 20, 20]], 'onnx_opset_version': 11}


def create_pytorch_nn_module_shapes_list_dynamic(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    inp_shapes = [[Dimension(-1), 3, 20, Dimension(20, -1)], [-1, 3, 20, Dimension(-1, 20)]]

    param1 = ov.opset8.parameter(PartialShape(inp_shapes[0]), name="input_0", dtype=np.float32)
    param2 = ov.opset8.parameter(PartialShape(inp_shapes[1]), name="input_1", dtype=np.float32)
    add = ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(add)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return pt_model, ref_model, {'input_shape': inp_shapes, 'onnx_opset_version': 11}


def create_pytorch_nn_module_shapes_list_dynamic_single_input(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shapes = [[Dimension(-1), 3, 20, Dimension(20, -1)]]
    ref_model = make_ref_pt_model_one_input(inp_shapes[0])
    return pt_model, ref_model, {'input_shape': inp_shapes, 'onnx_opset_version': 11}


def create_pytorch_nn_module_shapes_list_static_single_input(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shapes = [[1, 3, 20, 20]]
    ref_model = make_ref_pt_model_one_input(inp_shapes[0])
    return pt_model, ref_model, {'input_shape': inp_shapes, 'onnx_opset_version': 11}


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
        create_pytorch_nn_module_sample_input_dict,
        create_pytorch_nn_module_sample_input_dict_two_inputs,
        create_pytorch_nn_module_sample_list_of_tensors,
        create_pytorch_jit_script_module,
        create_pytorch_jit_script_function,
        create_pytorch_nn_module_layout_list,
        create_pytorch_nn_module_layout_list_case2,
        create_pytorch_nn_module_mean_list,
        create_pytorch_nn_module_scale_list,
        create_pytorch_nn_module_shapes_list_static,
        create_pytorch_nn_module_shapes_list_dynamic,
        create_pytorch_nn_module_shapes_list_dynamic_single_input,
        create_pytorch_nn_module_shapes_list_static_single_input
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
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)
