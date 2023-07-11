# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy
import numpy as np
import openvino.runtime as ov
import pytest
import torch
import unittest
from openvino.runtime import PartialShape, Dimension, Model, Type, InputCutInfo

from common.mo_convert_test_class import CommonMOConvertTest


class MyTorchOp(torch.autograd.Function):
    @staticmethod
    def symbolic(g, in_positions):
        return g.op("MyTorchOp", in_positions)

    @staticmethod
    def forward(self, in_positions):
        out_pos = in_positions.reshape(-1)
        return out_pos + 0.5


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
            logits = self.linear_relu_stack(x * y)
            return logits

    return NeuralNetwork()


def make_pt_model_with_optional_input():
    from torch import nn

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.ReLU(),
                nn.Sigmoid(),
            )

        def forward(self, x, y=None, z=None):
            if y is None:
                logits = self.linear_relu_stack(x + z)
            if z is None:
                logits = self.linear_relu_stack(x * y)
            return logits
    
    return NeuralNetwork()

def make_ref_pt_model_one_input(shape, dtype=np.float32):
    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape, name="input_0", dtype=dtype)
    relu = ov.opset8.relu(param1)
    if dtype != np.float32:
        relu = ov.opset8.convert(relu, np.float32)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1]
    model = Model([sigm], parameter_list, "test")
    return model


def make_ref_pt_model_two_inputs(shape, dtype=np.float32):
    if len(shape) == 2:
        param1 = ov.opset8.parameter(PartialShape(
            shape[0]), name="input_0", dtype=dtype)
        param2 = ov.opset8.parameter(PartialShape(
            shape[1]), name="input_1", dtype=dtype)
    else:
        shape = PartialShape(shape)
        param1 = ov.opset8.parameter(shape, name="input_0", dtype=dtype)
        param2 = ov.opset8.parameter(shape, name="input_1", dtype=dtype)
    mul = ov.opset8.multiply(param1, param2)
    relu = ov.opset8.relu(mul)    
    if dtype != np.float32:
        relu = ov.opset8.convert(relu, np.float32)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model = Model([sigm], parameter_list, "test")
    return model


def make_ref_pt_model_with_optional_inputs(shape, dtype=np.float32, z_exist=False):
    if len(shape) == 2:
        param1 = ov.opset8.parameter(PartialShape(
            shape[0]), name="input_0", dtype=dtype)
        param2 = ov.opset8.parameter(PartialShape(
            shape[1]), name="input_1", dtype=dtype)
    else:
        shape = PartialShape(shape)
        param1 = ov.opset8.parameter(shape, name="input_0", dtype=dtype)
        param2 = ov.opset8.parameter(shape, name="input_1", dtype=dtype)
    
    op = ov.opset8.multiply(param1, param2) if not z_exist else ov.opset8.add(param1, param2)
    relu = ov.opset8.relu(op)    
    if dtype != np.float32:
        relu = ov.opset8.convert(relu, np.float32)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    model = Model([sigm], parameter_list, "test")
    return model


def create_pytorch_nn_module_case1(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([-1, -1, -1, -1])

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = sample_input1, sample_input2

    return pt_model, ref_model, {'example_input': sample_input}


def create_pytorch_nn_module_case2(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([-1, 3, -1, -1])

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = sample_input1, sample_input2

    return pt_model, ref_model, {'input_shape': ["[?,3,?,?]", PartialShape([-1, 3, -1, -1])],
                                 'example_input': sample_input}


def create_pytorch_nn_module_with_scalar_input(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([[], [-1, 3, -1, -1]])

    sample_input1 = torch.tensor(0.66)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = sample_input1, sample_input2

    return pt_model, ref_model, {'input_shape': ["[]", PartialShape([-1, 3, -1, -1])],
                                 'example_input': sample_input}


def create_pytorch_nn_module_case3(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([-1, 3, -1, -1])

    sample_input1 = torch.zeros(1, 3, 10, 10)
    sample_input2 = torch.zeros(1, 3, 10, 10)
    sample_input = tuple([sample_input1, sample_input2])

    return pt_model, ref_model, {'input_shape': "[?,3,?,?],[?,3,?,?]",
                                 'example_input': sample_input}


def create_pytorch_nn_module_case4(tmp_dir):
    pt_model = make_pt_model_one_input()

    sample_input = torch.zeros(1, 3, 10, 10)

    ref_model = make_ref_pt_model_one_input(PartialShape([1, 3, 20, 20]))

    return pt_model, ref_model, {'example_input': sample_input, "input_shape": [1, 3, 20, 20]}


def create_pytorch_nn_module_case5(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shape = PartialShape([-1, 3, Dimension(2, -1), Dimension(-1, 10)])
    ref_model = make_ref_pt_model_one_input(inp_shape)

    sample_input = torch.zeros(3, 3, 10, 10)
    return pt_model, ref_model, {'example_input': sample_input,
                                 'input': (inp_shape, np.float32)}


def create_pytorch_nn_module_case6(tmp_dir):
    pt_model = make_pt_model_one_input()
    shape = PartialShape([1, 3, Dimension(2, -1), Dimension(-1, 10)])
    ref_model = make_ref_pt_model_one_input(shape)

    return pt_model, ref_model, {'input': (shape, np.float32)}


def create_pytorch_nn_module_case7(tmp_dir):
    pt_model = make_pt_model_one_input()

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)

    ref_model = make_ref_pt_model_one_input(PartialShape([1, 3, 20, 20]), dtype=np.int32)

    return pt_model, ref_model, {'example_input': sample_input, "input": ([1, 3, 20, 20], np.int32)}

def create_pytorch_nn_module_torch_size(tmp_dir):
    pt_model = make_pt_model_one_input()
    ref_model = make_ref_pt_model_one_input([1, 3, 2, 10])

    return pt_model, ref_model, {'input': (torch.Size([1, 3, 2, 10]), np.float32)}


def create_pytorch_nn_module_sample_input_int32(tmp_dir):
    pt_model = make_pt_model_one_input()
    shape = PartialShape([-1, 3, Dimension(2, -1), Dimension(-1, 10)])

    sample_input = torch.zeros(1, 3, 10, 10, dtype=torch.int32)

    ref_model = make_ref_pt_model_one_input(shape, dtype=np.int32)

    return pt_model, ref_model, {'example_input': sample_input,
                                 'input': (shape, np.int32)}


def create_pytorch_nn_module_sample_input_int32_two_inputs(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    inp_shapes = ["[?,3,?,?]", PartialShape([-1, 3, -1, -1])]

    sample_input1 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input2 = torch.zeros(1, 3, 10, 10, dtype=torch.int32)
    sample_input = sample_input1, sample_input2
    ref_model = make_ref_pt_model_two_inputs(
        [PartialShape([-1, 3, -1, -1]), inp_shapes[1]], dtype=np.int32)

    return pt_model, ref_model, {'input_shape': inp_shapes,
                                 'input': [np.int32, np.int32],
                                 'example_input': sample_input}


def create_pytorch_jit_script_module(tmp_dir):
    import torch

    net = make_pt_model_two_inputs()
    scripted_model = torch.jit.script(net)

    model_ref = make_ref_pt_model_two_inputs([1, 3, 5, 5])
    return scripted_model, model_ref, {'input': [([1, 3, 5, 5], np.float32), ([1, 3, 5, 5], np.float32)]}


def create_pytorch_jit_script_function(tmp_dir):
    import torch

    @torch.jit.script
    def scripted_fn(x: torch.Tensor, y: torch.Tensor):
        return torch.sigmoid(torch.relu(x * y))

    inp_shape = PartialShape([Dimension(1, -1), Dimension(-1, 5), 10])
    ref_model = make_ref_pt_model_two_inputs(inp_shape)
    return scripted_fn, ref_model, {'input': [(inp_shape), (inp_shape)]}



def create_pytorch_nn_module_layout_list(tmp_dir):
    from openvino.runtime import Layout
    pt_model = make_pt_model_two_inputs()
    shape = [1, 3, 10, 10]

    shape = PartialShape(shape)
    ref_model = make_ref_pt_model_two_inputs(shape)
    ref_model.inputs[0].node.layout = Layout('nchw')
    ref_model.inputs[1].node.layout = Layout('nhwc')

    return pt_model, ref_model, {
        'input_shape': [shape, shape], 'layout': ['nchw', Layout('nhwc')],
        }


def create_pytorch_nn_module_layout_list_case2(tmp_dir):
    from openvino.runtime import Layout
    pt_model = make_pt_model_two_inputs()
    shape = [1, 3, 10, 10]

    shape = PartialShape(shape)
    ref_model = make_ref_pt_model_two_inputs(shape)
    ref_model.inputs[0].node.layout = Layout('nchw')
    ref_model.inputs[1].node.layout = Layout('nhwc')

    return pt_model, ref_model, {
        'input_shape': [shape, shape], 'layout': ('nchw', Layout('nhwc'))}


def create_pytorch_nn_module_mean_list(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    shape = [1, 10, 10, 3]

    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape)
    param2 = ov.opset8.parameter(shape)
    const1 = ov.opset8.constant([[[[-0.0, -0.0, -0.0]]]], dtype=np.float32)
    const2 = ov.opset8.constant([[[[-0.0, -0.0, -0.0]]]], dtype=np.float32)
    add1 = ov.opset8.add(param1, const1)
    add2 = ov.opset8.add(param2, const2)
    mul = ov.opset8.multiply(add1, add2)
    relu = ov.opset8.relu(mul)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {
        'input_shape': [shape, shape], 'mean_values': [[0, 0, 0], [0, 0, 0]], 'compress_to_fp16': False}


def create_pytorch_nn_module_mean_list_default_no_compression(tmp_dir):
    # by default compression is disabled (same as setting 'compress_to_fp16': False)
    pt_model = make_pt_model_two_inputs()
    shape = [1, 10, 10, 3]

    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape)
    param2 = ov.opset8.parameter(shape)
    const1 = ov.opset8.constant([[[[-0.0, -0.0, -0.0]]]], dtype=np.float32)
    const2 = ov.opset8.constant([[[[-0.0, -0.0, -0.0]]]], dtype=np.float32)
    add1 = ov.opset8.add(param1, const1)
    add2 = ov.opset8.add(param2, const2)
    mul = ov.opset8.multiply(add1, add2)
    relu = ov.opset8.relu(mul)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {'input_shape': [shape, shape], 'mean_values': [[0, 0, 0], [0, 0, 0]]}


def create_pytorch_nn_module_mean_list_compression_enabled(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    shape = [1, 10, 10, 3]

    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape)
    param2 = ov.opset8.parameter(shape)
    const1 = ov.opset8.constant([[[[-0.0, -0.0, -0.0]]]], dtype=np.float32)
    const2 = ov.opset8.constant([[[[-0.0, -0.0, -0.0]]]], dtype=np.float32)
    add1 = ov.opset8.add(param1, const1)
    add2 = ov.opset8.add(param2, const2)
    mul = ov.opset8.multiply(add1, add2)
    relu = ov.opset8.relu(mul)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {
        'input_shape': [shape, shape], 'mean_values': [[0, 0, 0], [0, 0, 0]], 
        'compress_to_fp16': False}


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
    mul = ov.opset8.multiply(sub1, sub2)
    relu = ov.opset8.relu(mul)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {'input_shape': [shape, shape], 'scale_values': [[1, 1, 1], [1, 1, 1]], 'compress_to_fp16': False}


def create_pytorch_nn_module_scale_list_default_no_compression(tmp_dir):
    # by default compression is disabled (same as setting 'compress_to_fp16': False)
    pt_model = make_pt_model_two_inputs()
    shape = [1, 10, 10, 3]

    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape)
    param2 = ov.opset8.parameter(shape)
    const1 = ov.opset8.constant([[[[1, 1, 1]]]], dtype=np.float32)
    const2 = ov.opset8.constant([[[[1, 1, 1]]]], dtype=np.float32)
    sub1 = ov.opset8.multiply(param1, const1)
    sub2 = ov.opset8.multiply(param2, const2)
    mul = ov.opset8.multiply(sub1, sub2)
    relu = ov.opset8.relu(mul)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {'input_shape': [shape, shape], 'scale_values': [[1, 1, 1], [1, 1, 1]]}


def create_pytorch_nn_module_scale_list_compression_enabled(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    shape = [1, 10, 10, 3]

    shape = PartialShape(shape)
    param1 = ov.opset8.parameter(shape)
    param2 = ov.opset8.parameter(shape)
    const1 = ov.opset8.constant([[[[1, 1, 1]]]], dtype=np.float16)
    const1_decompressed = ov.opset8.convert(const1, destination_type=np.float32)
    const2 = ov.opset8.constant([[[[1, 1, 1]]]], dtype=np.float16)
    const2_decompressed = ov.opset8.convert(const2, destination_type=np.float32)
    mul1 = ov.opset8.multiply(param1, const1_decompressed)
    mul2 = ov.opset8.multiply(param2, const2_decompressed)
    mul3 = ov.opset8.multiply(mul1, mul2)
    relu = ov.opset8.relu(mul3)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")

    return pt_model, ref_model, {'input_shape': [shape, shape], 'scale_values': [[1, 1, 1], [1, 1, 1]],
                                 'compress_to_fp16': True}


def create_pytorch_nn_module_shapes_list_static(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([1, 3, 20, 20])

    return pt_model, ref_model, {'input_shape': [[1, 3, 20, 20], [1, 3, 20, 20]]}


def create_pytorch_nn_module_shapes_list_static_via_input(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    ref_model = make_ref_pt_model_two_inputs([1, 3, 20, 20])

    return pt_model, ref_model, {'input': [([1, 3, 20, 20], np.float32), ([1, 3, 20, 20], np.float32)]}


def create_pytorch_nn_module_shapes_list_dynamic(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    inp_shapes = [[Dimension(-1), 3, 20, Dimension(20, -1)],
                  [-1, 3, 20, Dimension(-1, 20)]]

    param1 = ov.opset8.parameter(PartialShape(
        inp_shapes[0]), name="x", dtype=np.float32)
    param2 = ov.opset8.parameter(PartialShape(
        inp_shapes[1]), name="y", dtype=np.float32)
    mul = ov.opset8.multiply(param1, param2)
    relu = ov.opset8.relu(mul)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return pt_model, ref_model, {'input_shape': inp_shapes}


def create_pytorch_nn_module_shapes_list_dynamic_via_input(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    inp_shapes = [[Dimension(-1), 3, 20, Dimension(20, -1)],
                  [-1, 3, 20, Dimension(-1, 20)]]

    param1 = ov.opset8.parameter(PartialShape(
        inp_shapes[0]), name="x", dtype=np.float32)
    param2 = ov.opset8.parameter(PartialShape(
        inp_shapes[1]), name="y", dtype=np.float32)
    mul = ov.opset8.multiply(param1, param2)
    relu = ov.opset8.relu(mul)
    sigm = ov.opset8.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return pt_model, ref_model, {'input': [(inp_shapes[0],), (inp_shapes[1],)]}


def create_pytorch_nn_module_shapes_list_dynamic_single_input(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shapes = [[Dimension(-1), 3, 20, Dimension(20, -1)]]
    ref_model = make_ref_pt_model_one_input(inp_shapes[0])
    return pt_model, ref_model, {'input_shape': inp_shapes}


def create_pytorch_nn_module_shapes_list_dynamic_single_input_via_input(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shapes = [Dimension(-1), 3, 20, Dimension(20, -1)]
    ref_model = make_ref_pt_model_one_input(inp_shapes)
    return pt_model, ref_model, {'input': InputCutInfo(shape=inp_shapes, type=np.float32)}


def create_pytorch_nn_module_shapes_list_static_single_input(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shapes = [[1, 3, 20, 20]]
    ref_model = make_ref_pt_model_one_input(inp_shapes[0])
    return pt_model, ref_model, {'input_shape': inp_shapes}


def create_pytorch_nn_module_shapes_list_static_single_input_via_input(tmp_dir):
    pt_model = make_pt_model_one_input()
    inp_shapes = [1, 3, 20, 20]
    ref_model = make_ref_pt_model_one_input(inp_shapes)
    return pt_model, ref_model, {'input': (inp_shapes, np.float32)}


def create_pytorch_nn_module_convert_pytorch_frontend1(tmp_dir):
    pt_model = make_pt_model_one_input()
    shape = [-1, -1, -1, -1]
    shape = PartialShape(shape)
    param = ov.opset10.parameter(shape)
    relu = ov.opset10.relu(param)
    sigm = ov.opset10.sigmoid(relu)

    parameter_list = [param]
    ref_model = Model([sigm], parameter_list, "test")
    return pt_model, ref_model, {
        "example_input": torch.zeros((1, 3, 10, 10)),
        'input': [InputCutInfo(shape=[-1, -1, -1, -1], type="f32")]
    }


def create_pytorch_nn_module_convert_pytorch_frontend2(tmp_dir):
    pt_model = make_pt_model_one_input()
    shape = [-1, -1, -1, -1]
    shape = PartialShape(shape)
    param = ov.opset10.parameter(shape, Type.i32)
    relu = ov.opset10.relu(param)
    convt = ov.opset10.convert(relu, "f32")
    sigm = ov.opset10.sigmoid(convt)

    parameter_list = [param]
    ref_model = Model([sigm], parameter_list, "test")
    return pt_model, ref_model, {
        "example_input": torch.zeros((1, 3, 10, 10), dtype=torch.int32),
        'input': [InputCutInfo(shape=[-1, -1, -1, -1], type="i32")]
    }


def create_pytorch_nn_module_convert_pytorch_frontend3(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    shape = [-1, -1, -1, -1]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape, dtype=np.float32)
    param2 = ov.opset10.parameter(shape, dtype=np.float32)
    mul = ov.opset10.multiply(param1, param2)
    relu = ov.opset10.relu(mul)
    sigm = ov.opset10.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return pt_model, ref_model, {
        "example_input": [torch.zeros((1, 3, 10, 10)), torch.ones((1, 3, 10, 10))],
        'input': [InputCutInfo(shape=[-1, -1, -1, -1], type="f32"), InputCutInfo(shape=[-1, -1, -1, -1], type="f32")]
    }


def create_pytorch_nn_module_convert_pytorch_frontend4(tmp_dir):
    pt_model = make_pt_model_two_inputs()
    shape = [-1, -1, -1, -1]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape, dtype=np.float32)
    param2 = ov.opset10.parameter(shape, dtype=np.float32)
    mul = ov.opset10.multiply(param1, param2)
    relu = ov.opset10.relu(mul)
    sigm = ov.opset10.sigmoid(relu)

    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return pt_model, ref_model, {
        "example_input": {"x": torch.zeros((1, 3, 10, 10), dtype=torch.float32), "y": torch.ones((1, 3, 10, 10), dtype=torch.float32)},
        'input': [InputCutInfo(shape=[-1, -1, -1, -1], type="f32"), InputCutInfo(shape=[-1, -1, -1, -1], type="f32")]
    }


def create_pytorch_jit_script_module_convert_pytorch_frontend(tmp_dir):
    import torch

    net = make_pt_model_two_inputs()
    scripted_model = torch.jit.script(net)
    shape = [-1, -1, -1, -1]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape, dtype=np.float32)
    param2 = ov.opset10.parameter(shape, dtype=np.float32)
    mul = ov.opset10.multiply(param1, param2)
    relu = ov.opset10.relu(mul)
    sigm = ov.opset10.sigmoid(relu)
    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return scripted_model, ref_model,  {
        "example_input": [torch.zeros((1, 3, 10, 10)), torch.ones((1, 3, 10, 10))]}


def create_pytorch_jit_trace_module_convert_pytorch_frontend(tmp_dir):
    import torch

    net = make_pt_model_two_inputs()
    example_input = [torch.zeros((1, 3, 10, 10)), torch.ones((1, 3, 10, 10))]
    scripted_model = torch.jit.trace(net, example_input)
    shape = [-1, -1, -1, -1]
    shape = PartialShape(shape)
    param1 = ov.opset10.parameter(shape, dtype=np.float32)
    param2 = ov.opset10.parameter(shape, dtype=np.float32)
    mul = ov.opset10.multiply(param1, param2)
    relu = ov.opset10.relu(mul)
    sigm = ov.opset10.sigmoid(relu)
    parameter_list = [param1, param2]
    ref_model = Model([sigm], parameter_list, "test")
    return scripted_model, ref_model,  {"example_input": example_input}


def create_pytorch_module_convert_pytorch_frontend_oob(tmp_dir):
    import torch
    import torch.nn.functional as F

    class ConvModel(torch.nn.Module):
        def __init__(self):
            super(ConvModel, self).__init__()
            self.weights = torch.rand([1, 3, 3, 3])

        def forward(self, x):
            return F.conv2d(x, self.weights)

    net = ConvModel()
    shape = PartialShape([-1, 3, -1, -1])
    param1 = ov.opset10.parameter(shape, dtype=np.float32)
    weights = ov.opset10.constant(net.weights.numpy(force=True))
    conv = ov.opset10.convolution(param1, weights, strides=[1, 1],
                                  pads_begin=[0, 0], pads_end=[0, 0],
                                  dilations=[1, 1])
    parameter_list = [param1]
    ref_model = Model([conv], parameter_list, "test")
    return net, ref_model, {}


def create_pytorch_module_with_optional_inputs_case1(tmp_dir):
    net = make_pt_model_with_optional_input()
    example_input = {"x": torch.zeros((1,3,10,10)), "y": torch.ones((1,3,10,10))}
    ref_model = make_ref_pt_model_with_optional_inputs([-1, -1, -1, -1])
    return net, ref_model, {"example_input": example_input}


def create_pytorch_module_with_optional_inputs_case2(tmp_dir):
    net = make_pt_model_with_optional_input()
    example_input = {"x": torch.zeros((1,3,10,10)), "z": torch.ones((1,3,10,10))}
    ref_model = make_ref_pt_model_with_optional_inputs([-1, -1, -1, -1], z_exist=True)
    return net, ref_model, {"example_input": example_input}


def create_pytorch_module_with_optional_inputs_case3(tmp_dir):
    net = make_pt_model_with_optional_input()
    example_input = {"x": torch.zeros((1,3,10,10)), "z": torch.ones((1,3,10,10))}
    ref_model = make_ref_pt_model_with_optional_inputs([3, 3, 3, 3], z_exist=True)
    return net, ref_model, {"example_input": example_input, "input_shape": [[3, 3, 3, 3], [3, 3, 3, 3]]}


def create_pytorch_module_with_optional_inputs_case4(tmp_dir):
    net = make_pt_model_with_optional_input()
    ref_model = make_ref_pt_model_with_optional_inputs([3, 3, 3, 3], z_exist=True)
    return net, ref_model, {"input": [("x", [3, 3, 3, 3]), ("z", [3, 3, 3, 3])]}


def create_pytorch_module_with_optional_inputs_case5(tmp_dir):
    net = make_pt_model_with_optional_input()
    ref_model = make_ref_pt_model_with_optional_inputs([1, 3, -1, -1], z_exist=True)
    return net, ref_model, {"input": ["x", "z"], "input_shape": [[1, 3, -1, -1], [1, 3, -1, -1]]}


class TestMoConvertPyTorch(CommonMOConvertTest):
    test_data = [
        create_pytorch_nn_module_case1,
        create_pytorch_nn_module_case2,
        create_pytorch_nn_module_case3,
        create_pytorch_nn_module_case4,
        create_pytorch_nn_module_case5,
        create_pytorch_nn_module_case6,
        create_pytorch_nn_module_case7,
        create_pytorch_nn_module_torch_size,
        create_pytorch_nn_module_sample_input_int32,
        create_pytorch_nn_module_sample_input_int32_two_inputs,
        create_pytorch_jit_script_module,
        create_pytorch_jit_script_function,
        create_pytorch_nn_module_layout_list,
        create_pytorch_nn_module_layout_list_case2,
        create_pytorch_nn_module_mean_list,
        create_pytorch_nn_module_mean_list_default_no_compression,
        create_pytorch_nn_module_mean_list_compression_enabled,
        create_pytorch_nn_module_scale_list,
        create_pytorch_nn_module_scale_list_default_no_compression,
        create_pytorch_nn_module_scale_list_compression_enabled,
        create_pytorch_nn_module_shapes_list_static,
        create_pytorch_nn_module_shapes_list_static_via_input,
        create_pytorch_nn_module_shapes_list_dynamic,
        create_pytorch_nn_module_shapes_list_dynamic_via_input,
        create_pytorch_nn_module_shapes_list_dynamic_single_input,
        create_pytorch_nn_module_shapes_list_static_single_input,
        create_pytorch_nn_module_shapes_list_dynamic_single_input_via_input,
        create_pytorch_nn_module_shapes_list_static_single_input_via_input,
        create_pytorch_nn_module_convert_pytorch_frontend1,
        create_pytorch_nn_module_convert_pytorch_frontend2,
        create_pytorch_nn_module_convert_pytorch_frontend3,
        create_pytorch_nn_module_convert_pytorch_frontend4,
        create_pytorch_jit_script_module_convert_pytorch_frontend,
        create_pytorch_jit_trace_module_convert_pytorch_frontend,
        create_pytorch_module_convert_pytorch_frontend_oob,
        create_pytorch_module_with_optional_inputs_case1,
        create_pytorch_module_with_optional_inputs_case2,
        create_pytorch_module_with_optional_inputs_case3,
        create_pytorch_module_with_optional_inputs_case4,
        create_pytorch_module_with_optional_inputs_case5,
        create_pytorch_nn_module_with_scalar_input,
    ]

    @ pytest.mark.parametrize("create_model", test_data)
    @ pytest.mark.nightly
    @ pytest.mark.precommit
    def test_mo_import_from_memory(self, create_model, ie_device, precision, ir_version,
                                   temp_dir, use_new_frontend, use_old_api):
        fw_model, graph_ref, mo_params = create_model(temp_dir)

        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)
        self._test_by_ref_graph(temp_dir, test_params,
                                graph_ref, compare_tensor_names=False)


def create_pt_model_with_custom_op():
    #
    #   Create PyTorch model with custom operation
    #
    import torch.nn as nn

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.my_op = MyTorchOp()

        def forward(self, x):
            return self.my_op.apply(x)

    return MyModel()


class ConvertRaises(unittest.TestCase):
    def test_example_inputs(self):
        from openvino.runtime import convert_model
        pytorch_model = create_pt_model_with_custom_op()

        # Check that mo raises error message of wrong argument.
        with self.assertRaisesRegex(AssertionError, ".*argument is not recognized.*"):
            convert_model(pytorch_model, example_inputs=(torch.tensor(1),))

    def test_failed_extension(self):
        from openvino.tools.mo import convert_model
        from openvino.frontend.pytorch import ConversionExtension

        inp_shapes = [1, 3, 20, 20]
        pt_model = make_pt_model_one_input()

        def relu_bad(n):
            assert False, "Something happened"

        # Check that mo raises error message of wrong argument.
        with self.assertRaisesRegex(Exception, ".*Conversion is failed for: aten::relu.*"):
            convert_model(pt_model, input=(inp_shapes, np.float32), extensions=[
                          ConversionExtension("aten::relu", relu_bad)])
