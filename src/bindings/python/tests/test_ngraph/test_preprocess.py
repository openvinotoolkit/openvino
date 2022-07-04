# -*- coding: utf-8 -*-
# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime as ov
import openvino.runtime.opset8 as ops
from openvino.runtime import Model, Output, Type
from openvino.runtime.utils.decorators import custom_preprocess_function
from openvino.runtime import Core
from tests.runtime import get_runtime
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm


def test_ngraph_preprocess_mean():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    prep = inp.preprocess()
    prep.mean(1.0)
    function = ppp.build()

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
    function = Model(model, [parameter_a], "TestFunction")
    layout = ov.Layout("NC")

    ppp = PrePostProcessor(function)
    ppp.input().tensor().set_layout(layout)
    ppp.input().preprocess().mean([1., 2.])
    function = ppp.build()

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
    function = Model(model, [parameter_a], "TestFunction")
    layout = ov.Layout("NC")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout)
    inp.preprocess().scale([0.5, 2.0])
    function = ppp.build()

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
    function = Model([param1, param2], [param1, param2], "TestFunction")

    @custom_preprocess_function
    def custom_preprocess(output: Output):
        return ops.abs(output)

    ppp = PrePostProcessor(function)
    inp2 = ppp.input(1)
    inp2.tensor().set_element_type(Type.i32)
    inp2.preprocess().convert_element_type(Type.f32).mean(1.).scale(2.)
    inp2.preprocess().convert_element_type()
    inp1 = ppp.input(0)
    inp1.preprocess().convert_element_type(Type.f32).mean(1.).custom(custom_preprocess)
    function = ppp.build()

    input_data1 = np.array([[0, 1], [2, -2]]).astype(np.int32)
    input_data2 = np.array([[1, 3], [5, 7]]).astype(np.int32)
    expected_output1 = np.array([[1, 0], [1, 3]]).astype(np.float32)
    expected_output2 = np.array([[0, 1], [2, 3]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    [output1, output2] = computation(input_data1, input_data2)
    assert np.equal(output1, expected_output1).all()
    assert np.equal(output2, expected_output2).all()


def test_ngraph_preprocess_input_output_by_name():
    shape = [2, 2]
    param1 = ops.parameter(shape, dtype=np.int32, name="A")
    param2 = ops.parameter(shape, dtype=np.int32, name="B")
    function = Model([param1, param2], [param1, param2], "TestFunction")

    @custom_preprocess_function
    def custom_preprocess(output: Output):
        return ops.abs(output)

    ppp = PrePostProcessor(function)
    inp2 = ppp.input("B")
    inp2.tensor().set_element_type(Type.i32)
    inp2.preprocess().convert_element_type(Type.f32).mean(1.).scale(2.)
    inp1 = ppp.input("A")
    inp1.preprocess().convert_element_type(Type.f32).mean(1.)
    out1 = ppp.output("A")
    out1.postprocess().custom(custom_preprocess)
    out2 = ppp.output("B")
    out2.postprocess().custom(custom_preprocess)
    function = ppp.build()

    input_data1 = np.array([[0, 1], [2, -2]]).astype(np.int32)
    input_data2 = np.array([[-1, 3], [5, 7]]).astype(np.int32)
    expected_output1 = np.array([[1, 0], [1, 3]]).astype(np.float32)
    expected_output2 = np.array([[1, 1], [2, 3]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    [output1, output2] = computation(input_data1, input_data2)
    assert np.equal(output1, expected_output1).all()
    assert np.equal(output2, expected_output2).all()


def test_ngraph_preprocess_output_postprocess():
    shape = [2, 3]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NC")
    layout2 = ov.Layout("CN")
    layout3 = [1, 0]

    @custom_preprocess_function
    def custom_postprocess(output: Output):
        return ops.abs(output)

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout1)
    inp.preprocess().convert_element_type(Type.f32).mean([1.0, 2.0, 3.0])
    out = ppp.output()
    out.tensor().set_element_type(Type.f32)
    out.model().set_layout(layout1)
    out.postprocess().convert_element_type(Type.f32)
    out.postprocess().convert_layout(layout2).convert_layout(layout3)
    out.postprocess().custom(custom_postprocess).convert_element_type(Type.f16).convert_element_type()
    function = ppp.build()

    input_data = np.array([[-1, -2, -3], [-4, -5, -6]]).astype(np.int32)
    expected_output = np.array([[2, 4, 6], [5, 7, 9]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_spatial_static_shape():
    shape = [2, 2, 2]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout = ov.Layout("CHW")

    color_format = ColorFormat.RGB

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout).set_spatial_static_shape(2, 2).set_color_format(color_format)
    inp.preprocess().convert_element_type(Type.f32).mean([1., 2.])
    inp.model().set_layout(layout)
    out = ppp.output()
    out.tensor().set_layout(layout).set_element_type(Type.f32)
    out.model().set_layout(layout)
    function = ppp.build()

    input_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.int32)
    expected_output = np.array([[[0, 1], [2, 3]], [[3, 4], [5, 6]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_set_shape():
    shape = [1, 1, 1]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    @custom_preprocess_function
    def custom_crop(out_node: Output):
        start = ops.constant(np.array([1, 1, 1]), dtype=np.int32)
        stop = ops.constant(np.array([2, 2, 2]), dtype=np.int32)
        step = ops.constant(np.array([1, 1, 1]), dtype=np.int32)
        axis = ops.constant(np.array([0, 1, 2]), dtype=np.int32)
        return ops.slice(out_node, start, stop, step, axis)

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_shape([3, 3, 3])
    inp.preprocess().custom(custom_crop)
    function = ppp.build()

    input_data = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                           [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                           [[18, 19, 20], [21, 22, 23], [24, 25, 26]]]).astype(np.int32)
    expected_output = np.array([[[13]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_set_from_tensor():
    shape = [1, 224, 224, 3]
    inp_shape = [1, 480, 640, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_a.set_layout(ov.Layout("NHWC"))
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    input_data = ov.Tensor(Type.i32, inp_shape)
    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_from(input_data)
    inp.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    function = ppp.build()
    assert function.input().shape == ov.Shape(inp_shape)
    assert function.input().element_type == Type.i32
    assert function.output().shape == ov.Shape(shape)
    assert function.output().element_type == Type.f32


def test_ngraph_preprocess_set_from_np_infer():
    shape = [1, 1, 1]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    @custom_preprocess_function
    def custom_crop(out_node: Output):
        start = ops.constant(np.array([1, 1, 1]), dtype=np.int32)
        stop = ops.constant(np.array([2, 2, 2]), dtype=np.int32)
        step = ops.constant(np.array([1, 1, 1]), dtype=np.int32)
        axis = ops.constant(np.array([0, 1, 2]), dtype=np.int32)
        return ops.slice(out_node, start, stop, step, axis)

    input_data = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                           [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
                           [[18, 19, 20], [21, 22, 23], [24, 25, 26]]]).astype(np.int32)

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_from(input_data)
    inp.preprocess().convert_element_type().custom(custom_crop)
    function = ppp.build()
    assert function.input().shape == ov.Shape([3, 3, 3])
    assert function.input().element_type == Type.i32

    expected_output = np.array([[[13]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_set_memory_type():
    shape = [1, 1, 1]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    op = ops.relu(parameter_a)
    model = op
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    ppp.input().tensor().set_memory_type("some_memory_type")
    function = ppp.build()

    assert any(key for key in function.input().rt_info if "memory_type" in key)


@pytest.mark.parametrize(
    ("algorithm", "color_format1", "color_format2", "is_failing"),
    [(ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.UNDEFINED, ColorFormat.BGR, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.RGB, ColorFormat.I420_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.RGB, ColorFormat.I420_THREE_PLANES, True),
     (ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.RGB, ColorFormat.NV12_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.RGB, ColorFormat.RGBX, True),
     (ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.RGB, ColorFormat.BGRX, True),
     (ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.RGB, ColorFormat.NV12_TWO_PLANES, True),
     (ResizeAlgorithm.RESIZE_LINEAR, ColorFormat.UNDEFINED, ColorFormat.I420_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.RGB, ColorFormat.UNDEFINED, True),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.RGB, ColorFormat.BGR, False),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.BGR, ColorFormat.RGB, False),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.BGR, ColorFormat.RGBX, True),
     (ResizeAlgorithm.RESIZE_CUBIC, ColorFormat.BGR, ColorFormat.BGRX, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.I420_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.I420_THREE_PLANES, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.NV12_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.NV12_TWO_PLANES, True),
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.UNDEFINED, True)])
def test_ngraph_preprocess_steps(algorithm, color_format1, color_format2, is_failing):
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NCWH")
    layout2 = ov.Layout("NCHW")

    custom_processor = PrePostProcessor(function)
    inp = custom_processor.input()
    inp.tensor().set_layout(layout1).set_color_format(color_format1, [])
    inp.preprocess().mean(1.).resize(algorithm, 3, 3)
    inp.preprocess().convert_layout(layout2).convert_color(color_format2)

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
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NCWH")
    layout2 = ov.Layout("NCHW")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout1)
    inp.preprocess().mean(1.).convert_layout(layout2).reverse_channels()
    out = ppp.output()
    out.postprocess().convert_layout([0, 1, 2, 3])
    function = ppp.build()

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
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = ov.Layout("NCWH")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout1)
    inp.preprocess().mean(1.).reverse_channels()
    function = ppp.build()

    input_data = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]]).astype(np.float32)
    expected_output = np.array([[[[4, 5], [6, 7]], [[0, 1], [2, 3]]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_crop():
    orig_shape = [1, 2, 1, 1]
    tensor_shape = [1, 2, 3, 3]
    parameter_a = ops.parameter(orig_shape, dtype=np.float32, name="A")
    model = ops.relu(parameter_a)
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    ppp.input().tensor().set_shape(tensor_shape)
    ppp.input().preprocess().crop([0, 0, 1, 1], [1, 2, -1, -1])
    function = ppp.build()

    input_data = np.arange(18).astype(np.float32).reshape(tensor_shape)
    expected_output = np.array([4, 13]).astype(np.float32).reshape(orig_shape)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data)
    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_resize_algorithm():
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    resize_alg = ResizeAlgorithm.RESIZE_CUBIC
    layout1 = ov.Layout("NCWH")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout1)
    inp.preprocess().mean(1.).resize(resize_alg, 3, 3)
    function = ppp.build()

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

    ppp = PrePostProcessor(function)
    ppp.input(1).preprocess().convert_element_type(Type.f32).scale(0.5)
    ppp.input(0).preprocess().convert_element_type(Type.f32).mean(5.)
    ppp.output(0).postprocess().custom(custom_preprocess)
    function = ppp.build()

    input_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]).astype(np.float32)
    expected_output = np.array([[[2, 1], [4, 7]], [[10, 13], [16, 19]]]).astype(np.float32)

    runtime = get_runtime()
    computation = runtime.computation(function)
    output = computation(input_data, input_data)

    assert np.equal(output, expected_output).all()


def test_ngraph_preprocess_dump():
    shape = [1, 3, 224, 224]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="RGB_input")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    ppp.input().tensor().set_layout(ov.Layout("NHWC")).set_element_type(Type.u8)
    ppp.input().tensor().set_spatial_dynamic_shape()
    ppp.input().preprocess().convert_element_type(Type.f32).reverse_channels()
    ppp.input().preprocess().mean([1, 2, 3]).scale([4, 5, 6])
    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    ppp.input().model().set_layout(ov.Layout("NCHW"))
    p_str = str(ppp)
    print(ppp)
    assert "Pre-processing steps (5):" in p_str
    assert "convert type (f32):" in p_str
    assert "reverse channels:" in p_str
    assert "mean (1,2,3):" in p_str
    assert "scale (4,5,6):" in p_str
    assert "resize to model width/height:" in p_str
    assert "Implicit pre-processing steps (1):" in p_str
    assert "convert layout " + ov.Layout("NCHW").to_string() in p_str
