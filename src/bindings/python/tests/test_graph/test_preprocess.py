# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import openvino.runtime.opset13 as ops

from openvino import Core, Layout, Model, Shape, Tensor, Type
from openvino.runtime.utils.decorators import custom_preprocess_function
from openvino.runtime import Output
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm


def test_graph_preprocess_mean():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    ppp = PrePostProcessor(function)
    inp = ppp.input()
    prep = inp.preprocess()
    prep.mean(1.0)
    function = ppp.build()
    model_operators = [op.get_name().split("_")[0] for op in function.get_ordered_ops()]
    assert len(model_operators) == 4
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [2, 2]
    assert function.get_output_element_type(0) == Type.f32
    assert "Constant" in model_operators
    assert "Subtract" in model_operators


def test_graph_preprocess_mean_vector():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout = Layout("NC")

    ppp = PrePostProcessor(function)
    ppp.input().tensor().set_layout(layout)
    ppp.input().preprocess().mean([1., 2.])
    function = ppp.build()

    model_operators = [op.get_name().split("_")[0] for op in function.get_ordered_ops()]
    assert len(model_operators) == 4
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [2, 2]
    assert function.get_output_element_type(0) == Type.f32
    assert "Constant" in model_operators
    assert "Subtract" in model_operators


def test_graph_preprocess_scale_vector():
    shape = [2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout = Layout("NC")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout)
    inp.preprocess().scale([0.5, 2.0])
    function = ppp.build()

    model_operators = [op.get_name().split("_")[0] for op in function.get_ordered_ops()]
    assert len(model_operators) == 4
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [2, 2]
    assert function.get_output_element_type(0) == Type.f32
    assert "Constant" in model_operators
    assert "Divide" in model_operators


def test_graph_preprocess_mean_scale_convert():
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

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Convert",
        "Constant",
        "Subtract",
        "Divide",
        "Result",
        "Abs",
    ]
    assert len(model_operators) == 15
    assert function.get_output_size() == 2
    assert list(function.get_output_shape(0)) == [2, 2]
    assert list(function.get_output_shape(1)) == [2, 2]
    assert function.get_output_element_type(0) == Type.i32
    assert function.get_output_element_type(1) == Type.i32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_input_output_by_name():
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

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Convert",
        "Constant",
        "Subtract",
        "Divide",
        "Result",
        "Abs",
    ]
    assert len(model_operators) == 16
    assert function.get_output_size() == 2
    assert list(function.get_output_shape(0)) == [2, 2]
    assert list(function.get_output_shape(1)) == [2, 2]
    assert function.get_output_element_type(0) == Type.i32
    assert function.get_output_element_type(1) == Type.i32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_output_postprocess():
    shape = [2, 3]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = Layout("NC")
    layout2 = Layout("CN")
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

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Convert",
        "Constant",
        "Subtract",
        "Transpose",
        "Result",
        "Abs",
    ]
    assert len(model_operators) == 14
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [2, 3]
    assert function.get_output_element_type(0) == Type.f32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_spatial_static_shape():
    shape = [3, 2, 2]
    parameter_a = ops.parameter(shape, dtype=np.int32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout = Layout("CHW")

    color_format = ColorFormat.RGB

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout).set_spatial_static_shape(2, 2).set_color_format(color_format)
    inp.preprocess().convert_element_type(Type.f32).mean([1., 2., 3.])
    inp.model().set_layout(layout)
    out = ppp.output()
    out.tensor().set_layout(layout).set_element_type(Type.f32)
    out.model().set_layout(layout)
    function = ppp.build()

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Convert",
        "Constant",
        "Subtract",
        "Result",
    ]
    assert len(model_operators) == 7
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [3, 2, 2]
    assert function.get_output_element_type(0) == Type.f32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_set_shape():
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

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Constant",
        "Result",
        "Slice",
    ]
    assert len(model_operators) == 7
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [1, 1, 1]
    assert function.get_output_element_type(0) == Type.i32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_set_from_tensor():
    shape = [1, 224, 224, 3]
    inp_shape = [1, 480, 640, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    parameter_a.set_layout(Layout("NHWC"))
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    input_data = Tensor(Type.i32, inp_shape)
    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_from(input_data)
    inp.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    function = ppp.build()
    assert function.input().shape == Shape(inp_shape)
    assert function.input().element_type == Type.i32
    assert function.output().shape == Shape(shape)
    assert function.output().element_type == Type.f32


def test_graph_preprocess_set_from_np_infer():
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
    assert function.input().shape == Shape([3, 3, 3])
    assert function.input().element_type == Type.i32

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Convert",
        "Constant",
        "Result",
        "Slice",
    ]
    assert len(model_operators) == 8
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [1, 1, 1]
    assert function.get_output_element_type(0) == Type.f32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_set_memory_type():
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
     (ResizeAlgorithm.RESIZE_NEAREST, ColorFormat.BGR, ColorFormat.UNDEFINED, True),
     (ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, ColorFormat.UNDEFINED, ColorFormat.BGR, True),
     (ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, ColorFormat.RGB, ColorFormat.NV12_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, ColorFormat.RGB, ColorFormat.RGBX, True),
     (ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, ColorFormat.RGB, ColorFormat.BGRX, True),
     (ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, ColorFormat.RGB, ColorFormat.NV12_TWO_PLANES, True),
     (ResizeAlgorithm.RESIZE_BILINEAR_PILLOW, ColorFormat.UNDEFINED, ColorFormat.I420_SINGLE_PLANE, True),
     (ResizeAlgorithm.RESIZE_BICUBIC_PILLOW, ColorFormat.RGB, ColorFormat.UNDEFINED, True),
     (ResizeAlgorithm.RESIZE_BICUBIC_PILLOW, ColorFormat.RGB, ColorFormat.BGR, False),
     (ResizeAlgorithm.RESIZE_BICUBIC_PILLOW, ColorFormat.BGR, ColorFormat.RGB, False),
     (ResizeAlgorithm.RESIZE_BICUBIC_PILLOW, ColorFormat.BGR, ColorFormat.RGBX, True),
     (ResizeAlgorithm.RESIZE_BICUBIC_PILLOW, ColorFormat.BGR, ColorFormat.BGRX, True),
     ])
def test_graph_preprocess_steps(algorithm, color_format1, color_format2, is_failing):
    shape = [1, 3, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = Layout("NCWH")
    layout2 = Layout("NCHW")

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
        model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
        expected_ops = [
            "Parameter",
            "Constant",
            "Result",
            "Gather",
            "Interpolate",
        ]
        assert len(model_operators) == 15
        assert function.get_output_size() == 1
        assert list(function.get_output_shape(0)) == [1, 3, 3, 3]
        assert function.get_output_element_type(0) == Type.f32
        for op in expected_ops:
            assert op in model_operators


@pytest.mark.parametrize(
    ("color_format1", "color_format2", "tensor_in_shape", "model_in_shape"),
    [(ColorFormat.RGB, ColorFormat.GRAY, [1, 3, 3, 3], [1, 3, 3, 1]),
     (ColorFormat.BGR, ColorFormat.GRAY, [1, 3, 3, 3], [1, 3, 3, 1]),
     ])
def test_graph_preprocess_convert_color(color_format1, color_format2, tensor_in_shape, model_in_shape):
    parameter_a = ops.parameter(model_in_shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    custom_processor = PrePostProcessor(function)
    inp = custom_processor.input()
    inp.tensor().set_color_format(color_format1)
    inp.preprocess().convert_color(color_format2)
    function = custom_processor.build()

    assert function.get_output_size() == 1
    assert list(function.inputs[0].shape) == tensor_in_shape
    assert list(function.get_output_shape(0)) == model_in_shape
    assert function.get_output_element_type(0) == Type.f32


def test_graph_preprocess_postprocess_layout():
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = Layout("NCWH")
    layout2 = Layout("NCHW")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout1)
    inp.preprocess().mean(1.).convert_layout(layout2).reverse_channels()
    out = ppp.output()
    out.postprocess().convert_layout([0, 1, 2, 3])
    function = ppp.build()

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Constant",
        "Result",
        "Gather",
        "Range",
        "Transpose",
    ]
    assert len(model_operators) == 14
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [1, 1, 3, 3]
    assert function.get_output_element_type(0) == Type.f32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_reverse_channels():
    shape = [1, 2, 2, 2]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    layout1 = Layout("NCWH")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout1)
    inp.preprocess().mean(1.).reverse_channels()
    function = ppp.build()

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Constant",
        "Result",
        "Gather",
        "Range",
    ]
    assert len(model_operators) == 10
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [1, 2, 2, 2]
    assert function.get_output_element_type(0) == Type.f32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_crop():
    orig_shape = [1, 2, 1, 1]
    tensor_shape = [1, 2, 3, 3]
    parameter_a = ops.parameter(orig_shape, dtype=np.float32, name="A")
    model = ops.relu(parameter_a)
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    ppp.input().tensor().set_shape(tensor_shape)
    ppp.input().preprocess().crop([0, 0, 1, 1], [1, 2, -1, -1])
    function = ppp.build()

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Constant",
        "Result",
        "Relu",
        "Slice",
    ]
    assert len(model_operators) == 8
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [1, 2, 1, 1]
    assert function.get_output_element_type(0) == Type.f32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_resize_algorithm():
    shape = [1, 1, 3, 3]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="A")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")
    resize_alg = ResizeAlgorithm.RESIZE_CUBIC
    layout1 = Layout("NCWH")

    ppp = PrePostProcessor(function)
    inp = ppp.input()
    inp.tensor().set_layout(layout1)
    inp.preprocess().mean(1.).resize(resize_alg, 3, 3)
    function = ppp.build()

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Constant",
        "Result",
        "Subtract",
        "Interpolate",
    ]
    assert len(model_operators) == 7
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [1, 1, 3, 3]
    assert function.get_output_element_type(0) == Type.f32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_model():
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

    model_operators = [op.get_name().split("_")[0] for op in function.get_ops()]
    expected_ops = [
        "Parameter",
        "Constant",
        "Result",
        "Subtract",
        "Convert",
        "Abs",
        "Add",
        "Divide",
    ]
    assert len(model_operators) == 13
    assert function.get_output_size() == 1
    assert list(function.get_output_shape(0)) == [2, 2, 2]
    assert function.get_output_element_type(0) == Type.i32
    for op in expected_ops:
        assert op in model_operators


def test_graph_preprocess_dump():
    shape = [1, 3, 224, 224]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="RGB_input")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    ppp.input().tensor().set_layout(Layout("NHWC")).set_element_type(Type.u8)
    ppp.input().tensor().set_spatial_dynamic_shape()
    ppp.input().preprocess().convert_element_type(Type.f32).reverse_channels()
    ppp.input().preprocess().mean([1, 2, 3]).scale([4, 5, 6])
    ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
    ppp.input().model().set_layout(Layout("NCHW"))
    p_str = str(ppp)
    assert "Pre-processing steps (5):" in p_str
    assert "convert type (f32):" in p_str
    assert "reverse channels:" in p_str
    assert "mean (1,2,3):" in p_str
    assert "scale (4,5,6):" in p_str
    assert "resize to model width/height:" in p_str
    assert "Implicit pre-processing steps (1):" in p_str
    assert "convert layout " + Layout("NCHW").to_string() in p_str


@pytest.mark.parametrize(
    ("layout", "layout_str"),
    [("NHCW", "[N,H,C,W]"), ("NHWC", "[N,H,W,C]")])
def test_ngraph_set_layout_by_string(layout, layout_str):
    shape = [1, 3, 224, 224]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="RGB_input")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    ppp.input().model().set_layout(layout)
    p_str = str(ppp)
    assert f"{layout_str}" in p_str


@pytest.mark.parametrize(
    ("layout", "layout_str"),
    [(Layout("NHCW"), "[N,H,C,W]"), (Layout("NHWC"), "[N,H,W,C]")])
def test_ngraph_set_layout_by_layout_class(layout, layout_str):
    shape = [1, 3, 224, 224]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="RGB_input")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)
    ppp.input().model().set_layout(layout)
    p_str = str(ppp)
    assert f"{layout_str}" in p_str


@pytest.mark.parametrize(
    ("layout"),
    [("1-2-3D"), ("5-5")])
def test_ngraph_set_layout_by_str_thow_exception(layout):
    shape = [1, 3, 224, 224]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="RGB_input")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)

    with pytest.raises(RuntimeError) as e:
        ppp.input().model().set_layout(layout)
    assert "Layout name is invalid" in str(e.value)


def test_ngraph_set_layout_by_layout_class_thow_exception():
    shape = [1, 3, 224, 224]
    parameter_a = ops.parameter(shape, dtype=np.float32, name="RGB_input")
    model = parameter_a
    function = Model(model, [parameter_a], "TestFunction")

    ppp = PrePostProcessor(function)

    with pytest.raises(RuntimeError) as e:
        layout = Layout("1-2-3D")
        ppp.input().model().set_layout(layout)
    assert "Layout name is invalid" in str(e.value)
