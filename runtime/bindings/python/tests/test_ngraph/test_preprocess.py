# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino as ov
import openvino.opset8 as ops
from openvino.impl.preprocess import PrePostProcessor, InputInfo, PreProcessSteps, InputTensorInfo
from openvino.impl import Function, Output, Node, Type
from openvino.utils.decorators import custom_preprocess_function
from tests.runtime import get_runtime


def test_ngraph_preprocess_mean():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")

    function = PrePostProcessor()\
        .input(InputInfo()
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           )
               )\
        .build(function)

    input_data = np.array([[1, 2], [3, 4]]).astype(np.float32)
    expected_output = np.array([[0, 1], [2, 3]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()

def test_ngraph_preprocess_mean_vector():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    layout = ov.Layout("NCHW")

    function = PrePostProcessor()\
        .input(InputInfo()
               .tensor(InputTensorInfo().set_layout(layout))
               .preprocess(PreProcessSteps()
                           .mean([1., 2.])
                           )
               )\
        .build(function)

    input_data = np.array([[1, 2], [3, 4]]).astype(np.float32)
    expected_output = np.array([[0, 0], [2, 2]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_scale_vector():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    layout = ov.Layout("NCHW")

    function = PrePostProcessor()\
        .input(InputInfo()
               .tensor(InputTensorInfo().set_layout(layout))
               .preprocess(PreProcessSteps()
                           .scale([0.5, 2.])
                           )
               )\
        .build(function)

    input_data = np.array([[1, 2], [3, 4]]).astype(np.float32)
    expected_output = np.array([[2, 1], [6, 2]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_mean_scale_convert():
    shape = [2, 2]
    param1 = ops.parameter(shape, dtype=np.int32, name="A")
    param2 = ops.parameter(shape, dtype=np.int32, name="B")
    function = Function([param1, param2], [param1, param2], "TestFunction")

    @custom_preprocess_function
    def custom_preprocess(output: Output):
        return ops.abs(output)

    function = PrePostProcessor() \
        .input(InputInfo(1)
               .tensor(InputTensorInfo()
                       .set_element_type(Type.i32))
               .preprocess(PreProcessSteps()
                           .convert_element_type(Type.f32)
                           .mean(1.)
                           .scale(2.)
                           )
               ) \
        .input(InputInfo(0)
               .preprocess(PreProcessSteps()
                           .convert_element_type(Type.f32)
                           .mean(1.)
                           .custom(custom_preprocess)
                           )
               ) \
        .build(function)

    input_data1 = np.array([[0,1], [2,-2]]).astype(np.int32)
    input_data2 = np.array([[1,3], [5,7]]).astype(np.int32)
    expected_output1 = np.array([[1,0], [1,3]]).astype(np.float32)
    expected_output2 = np.array([[0,1], [2,3]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    [output1, output2] = computation(input_data1, input_data2)
    assert np.equal(output1, expected_output1).all()
    assert np.equal(output2, expected_output2).all()
