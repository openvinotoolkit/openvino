# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino as ov
import openvino.opset8 as ops
from openvino.impl import Function, Output, Type
from openvino.utils.decorators import custom_preprocess_function
from openvino import Core
from tests.runtime import get_runtime
from openvino.preprocess import PrePostProcessor, InputInfo, PreProcessSteps, InputTensorInfo, \
    OutputTensorInfo, OutputNetworkInfo, InputNetworkInfo, ColorFormat, OutputInfo, \
    PostProcessSteps, ResizeAlgorithm


def test_ngraph_preprocess_mean():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")

    function = PrePostProcessor(function)\
        .input(InputInfo()
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           )
               )\
        .build()

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

    function = PrePostProcessor(function)\
        .input(InputInfo()
               .tensor(InputTensorInfo().set_layout(layout))
               .preprocess(PreProcessSteps()
                           .mean([1., 2.])
                           )
               )\
        .build()

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

    function = PrePostProcessor(function)\
        .input(InputInfo()
               .tensor(InputTensorInfo().set_layout(layout))
               .preprocess(PreProcessSteps()
                           .scale([0.5, 2.])
                           )
               )\
        .build()

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

    function = PrePostProcessor(function) \
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
        .build()

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

    function = PrePostProcessor(function)\
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
        .build()

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

    color_format = ColorFormat.RGB
    function = PrePostProcessor(function)\
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
        .build()

    input_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    expected_output = np.array([[[0, 1], [2, 3]], [[3, 4], [5, 6]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


@pytest.mark.parametrize(
    "algorithm, color_format1, color_format2, is_failing",
    [(ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.UNDEFINED, ColorFormat.BGR, True),
     (ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.RGB, ColorFormat.NV12_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.RGB, ColorFormat.NV12_TWO_PLANES, True),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.RGB, ColorFormat.UNDEFINED, True),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.RGB, ColorFormat.BGR, False),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.BGR, ColorFormat.RGB, False),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.NV12_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.NV12_TWO_PLANES, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.UNDEFINED, True)])
def test_ngraph_preprocess_steps(algorithm, color_format1, color_format2, is_failing):
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NCWH")
    layout2 = ov.Layout("NCHW")
    custom_processor = PrePostProcessor(function)\
        .input(InputInfo()
               .tensor(InputTensorInfo()
                       .set_layout(layout1)
                       .set_color_format(color_format1, []))
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           .resize(algorithm, 3, 3)
                           .convert_layout(layout2)
                           .convert_color(color_format2)))
    if is_failing:
        with pytest.raises(RuntimeError) as e:
            function = custom_processor.build()
        assert "is not convertible to" in str(e.value)
    else:
        function = custom_processor.build()
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

    function = PrePostProcessor(function)\
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
        .build()

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

    function = PrePostProcessor(function)\
        .input(InputInfo()
               .tensor(InputTensorInfo()
                       .set_layout(layout1))
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           .reverse_channels()
                           )
               ) \
        .build()

    input_data = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    expected_output = np.array([[[[4, 5], [6, 7]], [[0, 1], [2, 3]]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_resize_algorithm():
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Function(model, [parameter_a], "TestFunction")
    resize_alg = ResizeAlgorithm.RESIZE_CUBIC
    layout1 = ov.Layout("NCWH")

    function = PrePostProcessor(function)\
        .input(InputInfo()
               .tensor(InputTensorInfo()
                       .set_layout(layout1))
               .preprocess(PreProcessSteps()
                           .mean(1.)
                           .resize(resize_alg, 3, 3))
               )\
        .build()

    input_data = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).astype(np.float32)
    expected_output = np.array([[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_model():
    model = bytes(b"""<net name="add_model" version="10">
    <layers>
    <layer id="0" name="x" type="Parameter" version="opset1">
        <data element_type="i32" shape="2,2,2"/>
        <output>
            <port id="0" precision="FP32">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
    <layer id="1" name="y" type="Parameter" version="opset1">
        <data element_type="i32" shape="2,2,2"/>
        <output>
            <port id="0" precision="FP32">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
    <layer id="2" name="sum" type="Add" version="opset1">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
            <port id="1">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="FP32">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </output>
    </layer>
    <layer id="3" name="sum/sink_port_0" type="Result" version="opset1">
        <input>
            <port id="0">
                <dim>2</dim>
                <dim>2</dim>
                <dim>2</dim>
            </port>
        </input>
    </layer>
    </layers>
    <edges>
    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
    <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>""")
    core = Core()
    function = core.read_model(model=model)

    @custom_preprocess_function
    def custom_preprocess(output: Output):
        return ops.abs(output)

    function = PrePostProcessor(function) \
        .input(InputInfo(1)
               .preprocess(PreProcessSteps()
                           .convert_element_type(Type.f32)
                           .scale(0.5)
                           )
               ) \
        .input(InputInfo(0)
               .preprocess(PreProcessSteps()
                           .convert_element_type(Type.f32)
                           .mean(5.))) \
        .output(OutputInfo(0)
                .postprocess(PostProcessSteps()
                             .custom(custom_preprocess))) \
        .build()

    input_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
    expected_output = np.array([[[2, 1], [4, 7]], [[10, 13], [16, 19]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data, input_data)

    assert np.equal(output, expected_output).all()
