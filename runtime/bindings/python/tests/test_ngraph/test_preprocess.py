# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino as ov
import openvino.opset8 as ops
from openvino.impl.preprocess import PrePostProcessor, InputInfo, PreProcessSteps, InputTensorInfo, \
    OutputTensorInfo, OutputNetworkInfo, InputNetworkInfo, ColorFormat, OutputInfo, \
    PostProcessSteps, ResizeAlgorithm
from openvino.impl import Function, Output, Type
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

    input_data1 = np.array([[0, 1], [2, -2]]).astype(np.int32)
    input_data2 = np.array([[1, 3], [5, 7]]).astype(np.int32)
    expected_output1 = np.array([[1, 0], [1, 3]]).astype(np.float32)
    expected_output2 = np.array([[0, 1], [2, 3]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    [output1, output2] = computation(input_data1, input_data2)
    assert np.equal(output1, expected_output1).all()
    assert np.equal(output2, expected_output2).all()


def test_ngraph_preprocess_output_postprocess():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NCHW")
    layout2 = ov.Layout("NHWC")
    layout3 = [0, 1]

    @custom_preprocess_function
    def custom_postprocess(output: Output):
        return ops.abs(output)

    function = PrePostProcessor()\
        .input(InputInfo()
               .tensor(InputTensorInfo().set_layout(layout1))
               .preprocess(PreProcessSteps()
                           .convert_element_type(Type.f32)
                           .mean([1., 2.])
                           )
               ) \
        .output(OutputInfo().postprocess(PostProcessSteps()
                                         .convert_element_type(Type.f32)
                                         .convert_layout(layout2)
                                         .convert_layout(layout3)
                                         .custom(custom_postprocess))) \
        .build(function)

    input_data = np.array([[-1, -2], [-3, -4]]).astype(np.int32)
    expected_output = np.array([[2, 4], [4, 6]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_spatial_static_shape():
    shape = [2, 2, 2]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    layout = ov.Layout("CHW")

    color_format = ColorFormat(3)
    function = PrePostProcessor()\
        .input(InputInfo()
               .tensor(InputTensorInfo()
                       .set_layout(layout)
                       .set_spatial_static_shape(2, 2)
                       .set_color_format(color_format, []))
               .preprocess(PreProcessSteps()
                           .convert_element_type(Type.f32)
                           .mean([1., 2])
                           )
               .network(InputNetworkInfo().set_layout(layout))
               ) \
        .output(OutputInfo()
                .tensor(OutputTensorInfo()
                        .set_layout(layout)
                        .set_element_type(Type.f32))
                .network(OutputNetworkInfo().set_layout(layout))) \
        .build(function)

    input_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    expected_output = np.array([[[0, 1], [2, 3]], [[3, 4], [5, 6]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_steps():
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    color_format1 = ColorFormat(3)
    color_format2 = ColorFormat(4)
    resize_alg = ResizeAlgorithm(0)
    layout1 = ov.Layout("NCWH")
    layout2 = ov.Layout("NCHW")

    function = PrePostProcessor()\
        .input(InputInfo()
               .tensor(InputTensorInfo()
                       .set_layout(layout1)
                       .set_color_format(color_format1, []))
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           .resize(resize_alg, 3, 3)
                           .convert_layout(layout2)
                           .convert_color(color_format2)
                           )
               )\
        .build(function)

    input_data = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).astype(np.float32)
    expected_output = np.array([[[[0, 3, 6], [1, 4, 7], [2, 5, 8]]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_postprocess_layout():
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NCWH")
    layout2 = ov.Layout("NCHW")

    function = PrePostProcessor()\
        .input(InputInfo()
               .tensor(InputTensorInfo()
                       .set_layout(layout1))
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           .convert_layout(layout2)
                           .reverse_channels()
                           )
               ) \
        .output(OutputInfo()
                .postprocess(PostProcessSteps()
                             .convert_layout([0, 1, 2, 3]))) \
        .build(function)

    input_data = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).astype(np.float32)
    expected_output = np.array([[[[0, 3, 6], [1, 4, 7], [2, 5, 8]]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_reverse_channels():
    shape = [1, 2, 2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NCWH")

    function = PrePostProcessor()\
        .input(InputInfo()
               .tensor(InputTensorInfo()
                       .set_layout(layout1))
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           .reverse_channels()
                           )
               ) \
        .build(function)

    input_data = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    expected_output = np.array([[[[4, 5], [6, 7]], [[0, 1], [2, 3]]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()
