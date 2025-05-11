# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import itertools
import math
import os
import re
import logging
from pathlib import Path

import torch
import numpy as np
import pytest

from openvino.frontend import FrontEndManager, ConversionExtension, NodeContext
from openvino import PartialShape, Type
import openvino.opset10 as ops

logging.basicConfig(level=logging.DEBUG)


class aten_relu(torch.nn.Module):
    def forward(self, x):
        return x, torch.nn.functional.relu(x)


class aten_multi_input_output(torch.nn.Module):
    def forward(self, x, y, z):
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        z = z.to(torch.float32)
        return torch.nn.functional.relu(x), x * y, z / x


def get_scripted_model(model):
    with torch.no_grad():
        model = torch.jit.script(model)
        model.eval()
        model = torch.jit.freeze(model)
        print(model.inlined_graph)  # will help debugging
        return model


def test_pytorch_fe_set_input_shape():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_partial_shape(place, PartialShape([1, 2, 3, 4]))
    om = fe.convert(im)
    assert om.get_parameters()[0].get_partial_shape(
    ) == PartialShape([1, 2, 3, 4])


def test_pytorch_fe_set_input_type():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_element_type(place, Type.f32)
    om = fe.convert(im)
    assert om.get_parameters()[0].get_element_type() == Type.f32


def test_pytorch_fe_set_input_value():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    place = im.get_place_by_tensor_name("x.1")
    im.set_partial_shape(place, PartialShape([1, 2, 3, 4]))
    im.set_element_type(place, Type.f32)
    im.set_tensor_value(place, np.random.randn(1, 2, 3, 4).astype(np.float32))
    om = fe.convert(im)
    assert len(om.get_parameters()) == 0


def test_pytorch_fe_override_all_inputs():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    decoder = TorchScriptPythonDecoder(aten_multi_input_output())
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    inputs = im.get_inputs()
    assert len(inputs) == 3, "Model should have 3 inputs."
    im.override_all_inputs([inputs[1], inputs[2], inputs[0]])
    om = fe.convert(im)
    assert om.get_parameters()[0].friendly_name == "y"
    assert om.get_parameters()[1].friendly_name == "z"
    assert om.get_parameters()[2].friendly_name == "x"


def test_pytorch_fe_removes_input_set_value():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    decoder = TorchScriptPythonDecoder(aten_multi_input_output())
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    assert len(im.get_inputs()) == 3, "Model should have 3 inputs."
    place = im.get_place_by_tensor_name("y")
    im.set_partial_shape(place, PartialShape([1]))
    im.set_element_type(place, Type.f32)
    im.set_tensor_value(place, np.random.randn(1).astype(np.float32))
    assert len(im.get_inputs()) == 2, "Model should have 2 inputs."
    om = fe.convert(im)
    assert om.get_parameters()[0].friendly_name == "x"
    assert om.get_parameters()[1].friendly_name == "z"


def test_conversion_extension():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()

        def forward(self, inp):
            elu = torch.nn.functional.elu(inp, alpha=0.123)
            gelu = torch.nn.functional.gelu(elu, approximate="none")
            gelu2 = torch.nn.functional.gelu(gelu, approximate="tanh")
            softmax = torch.nn.functional.softmax(gelu2, dim=-1)
            vn = torch.linalg.vector_norm(softmax, ord=math.inf, dim=None)
            return vn

    model = Model()
    decoder = TorchScriptPythonDecoder(get_scripted_model(model))

    def convert_elu(node: NodeContext):
        inp = node.get_input(0)
        alpha = node.get_input(1)
        zero = ops.constant(np.array([0], dtype=np.float32))
        greater = ops.greater(inp, zero)
        exp = ops.exp(inp)
        one = ops.constant(np.array([0], dtype=np.float32))
        sub = ops.subtract(exp, one)
        mul = ops.multiply(sub, alpha)
        select = ops.select(greater, inp, mul)
        return select.outputs()

    def convert_gelu(node: NodeContext):
        inp = node.get_input(0)
        approximate = node.get_values_from_const_input(1)
        if approximate == "none":
            f = ops.erf(ops.divide(inp, ops.constant(
                np.array([math.sqrt(2.0)], dtype=np.float32))))
        elif approximate == "tanh":
            f = ops.tanh(ops.multiply(ops.constant(np.array([math.sqrt(2.0 / math.pi)], dtype=np.float32)),
                                      ops.add(inp, ops.multiply(ops.constant(np.array([0.044715], dtype=np.float32)),
                                                                ops.power(inp, ops.constant(np.array([3], dtype=np.float32)))))))
        mul = ops.multiply(ops.multiply(ops.constant(np.array([0.5], dtype=np.float32)), inp),
                           ops.add(ops.constant(np.array([1], dtype=np.float32)), f))
        return mul.outputs()

    def convert_softmax(node: NodeContext):
        inp = node.get_input(0)
        dim = node.get_values_from_const_input(1, dtype=np.int32)
        dim_const = ops.constant(np.array([dim], dtype=np.int32))
        reduce_max = ops.reduce_max(inp, dim_const, True)
        sub = ops.subtract(inp, reduce_max)
        exp = ops.exp(sub)
        reduce_sum = ops.reduce_sum(exp, dim_const, True)
        div = ops.divide(exp, reduce_sum)
        return div.outputs()

    def convert_vector_norm(node: NodeContext):
        inp = node.get_input(0)
        ord = node.get_values_from_const_input(1)
        assert ord == math.inf
        dim = node.get_values_from_const_input(2)
        if dim is None:
            inp = ops.reshape(inp, ops.constant(np.array([-1])), False)
            reduce_axes = np.array([0])
        else:
            reduce_axes = np.array(dim)
        rm = ops.reduce_max(ops.abs(inp), reduce_axes, False)
        return rm.outputs()

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe
    fe.add_extension(ConversionExtension("aten::elu", convert_elu))
    fe.add_extension(ConversionExtension("aten::gelu", convert_gelu))
    fe.add_extension(ConversionExtension("aten::softmax", convert_softmax))
    fe.add_extension(ConversionExtension(
        "aten::linalg_vector_norm", convert_vector_norm))
    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "Constant", "Constant",
                                                                              "Constant", "Greater", "Exp",
                                                                              "Constant", "Subtract", "Constant",
                                                                              "Multiply", "Select", "Multiply",
                                                                              "Constant", "Constant", "Divide",
                                                                              "Erf", "Add", "Multiply",
                                                                              "Multiply", "Constant", "Constant",
                                                                              "Constant", "Constant", "Power",
                                                                              "Multiply", "Add", "Multiply",
                                                                              "Tanh", "Add", "Multiply",
                                                                              "Constant", "ReduceMax", "Subtract",
                                                                              "Exp", "ReduceSum", "Divide",
                                                                              "Constant", "Reshape", "Abs",
                                                                              "Constant", "ReduceMax", "Result"]


def get_builtin_extensions_path():
    base_paths = [Path(__file__).parent.parent.parent.parent]
    repo_dir = os.environ.get("REPO_DIR")
    if repo_dir:
        base_paths.append(repo_dir)

    for base_path in base_paths:
        paths = glob.glob(os.path.join(
            base_path, "**", "*test_builtin_extensions*"), recursive=True)
        for path in paths:
            if re.search(r"(lib)?test_builtin_extensions.?\.(dll|so)", path):
                return path
    raise RuntimeError("Unable to find test_builtin_extensions")


def test_so_extension():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    class Elu(torch.nn.Module):
        def __init__(self, alpha):
            super(Elu, self).__init__()
            self.alpha = alpha

        def forward(self, inp):
            return torch.nn.functional.elu(inp, self.alpha)

    model = Elu(alpha=0.123)
    decoder = TorchScriptPythonDecoder(get_scripted_model(model))

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe

    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        'Parameter', 'Elu', 'Constant', 'ConvertLike', 'Multiply', 'Result']

    fe.add_extension(get_builtin_extensions_path())
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "CustomElu", "Result"]


def test_framework_map_macros():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    class Relu(torch.nn.Module):
        def __init__(self):
            super(Relu, self).__init__()

        def forward(self, x):
            return torch.nn.functional.relu(x)

    model = Relu()
    decoder = TorchScriptPythonDecoder(get_scripted_model(model))

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe

    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Relu", "Result"]

    fe.add_extension(get_builtin_extensions_path())
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "ReluCustom", "Result"]


class CosModel(torch.nn.Module):
    def __init__(self):
        super(CosModel, self).__init__()

    def forward(self, x):
        return torch.cos(x.to(torch.float32))


def test_op_extension():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch import OpExtension

    model = CosModel()
    decoder = TorchScriptPythonDecoder(get_scripted_model(model))

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe

    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Convert", "Cos", "Result"]

    fe.add_extension(OpExtension("Sin", "aten::cos"))
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Convert", "Sin", "Result"]


def test_op_extension_generic():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino.frontend import OpExtension

    model = CosModel()
    decoder = TorchScriptPythonDecoder(get_scripted_model(model))

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe

    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Convert", "Cos", "Result"]

    fe.add_extension(OpExtension("Sin", "aten::cos"))
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Convert", "Sin", "Result"]


def test_module_extension():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch import ModuleExtension, ConversionExtension
    from openvino import convert_model

    class ModelWithModule(torch.nn.Module):
        def __init__(self):
            super(ModelWithModule, self).__init__()
            self.cos_module = CosModel()

        def forward(self, x):
            return self.cos_module(x)

    model = ModelWithModule()
    decoder = TorchScriptPythonDecoder(model)

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe

    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Convert", "Cos", "Result"]

    converted_model = convert_model(model, example_input=(
        torch.randn(100),), extension=[ModuleExtension(CosModel, "aten::sin")])
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Sin", "Result"]

    converted_model = convert_model(model, example_input=(torch.randn(
        100),), extension=[ModuleExtension(model.cos_module, "aten::sin")])
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Sin", "Result"]

    converted_model = convert_model(model, example_input=(torch.randn(
        100),), extension=[ModuleExtension("cos_module", "aten::sin")])
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Sin", "Result"]

    def sin_op(context):
        return ops.sin(context.get_input(0)).outputs()

    converted_model = convert_model(model, example_input=(torch.randn(100),), extension=[
                                    ModuleExtension("cos_module", "MyOp"), ConversionExtension("MyOp", sin_op)])
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Sin", "Result"]

    model = ModelWithModule()
    model.cos_module.flag = False
    me = ModuleExtension(CosModel,
                         "aten::sin",
                         condition=lambda m: getattr(m, "flag", False))
    converted_model = convert_model(model, example_input=(torch.randn(100),),
                                    extension=[me])
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Cos", "Result"]

    model = ModelWithModule()
    model.cos_module.flag = True
    converted_model = convert_model(model, example_input=(torch.randn(100),),
                                    extension=[me])
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Sin", "Result"]


def test_multiple_module_extension():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch import ModuleExtension
    from openvino import convert_model

    class ModelWithModule(torch.nn.Module):
        def __init__(self):
            super(ModelWithModule, self).__init__()
            self.cos_module = CosModel()
            self.relu_module = torch.nn.ReLU()

        def forward(self, x):
            x = x.to(torch.float32)
            return self.cos_module(x) + self.relu_module(x)

    model = ModelWithModule()
    decoder = TorchScriptPythonDecoder(model)

    fem = FrontEndManager()
    fe = fem.load_by_framework(framework="pytorch")
    assert fe

    input_model = fe.load(decoder)
    assert input_model
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Convert", "Convert", "Cos", "Constant", "Convert", "Relu", "Multiply", "Add", "Result"]

    converted_model = convert_model(model, example_input=(
        torch.randn(100),), extension=[ModuleExtension(CosModel, "aten::sin"), ModuleExtension(model.relu_module, "aten::tan")])
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == [
        "Parameter", "Sin", "Tan", "Add", "Result"]


def test_pytorch_telemetry():
    from openvino.frontend import TelemetryExtension
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    class MockTelemetry:
        def __init__(self, stat):
            self.stat = stat

        def send_event(self, *arg, **kwargs):
            self.stat["send_event"] += 1

        def send_error(self, *arg, **kwargs):
            self.stat["send_error"] += 1

        def send_stack_trace(self, *arg, **kwargs):
            self.stat["send_stack_trace"] += 1

    def add_ext(front_end, stat):
        tel = MockTelemetry(stat)
        front_end.add_extension(TelemetryExtension("mock",
                                                   tel.send_event,
                                                   tel.send_error,
                                                   tel.send_stack_trace))

    tel_stat = {"send_event": 0, "send_error": 0, "send_stack_trace": 0}
    # Ensure that MockTelemetry object is alive and can receive events (due to callbacks hold the object)
    model = get_scripted_model(aten_relu())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    add_ext(fe, tel_stat)
    im = fe.load(decoder)
    fe.convert(im)
    assert tel_stat["send_event"] == 2
    assert tel_stat["send_error"] == 0
    assert tel_stat["send_stack_trace"] == 0


class ShareWeghtsConvAndShareLinearModel(torch.nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.linear = torch.nn.Linear(4, 4)
        self.linear.weight.data = torch.randn((4, 4), dtype=torch.float32)
        self.linear.bias.data = torch.randn((1, 4), dtype=torch.float32)

    def forward(self, x):
        for _ in range(2):
            x = self.conv(x)
            x = self.linear(x)
        return x


def test_shared_consts_reused():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    model = ShareWeghtsConvAndShareLinearModel()
    decoder = TorchScriptPythonDecoder(
        model, example_input=(torch.rand(model.INPUT_SIZE),))
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    om = fe.convert(im)
    const_names = ["self.conv.weight",
                   "self.linear.weight", "self.linear.bias"]
    # self.conv.bias is not reused because of ConstantFolding
    for n in om.get_ops():
        if "Constant" in n.get_type_name():
            for name in n.output(0).names:
                if name in const_names:
                    const_names.remove(name)
                    assert len(n.output(0).get_target_inputs()
                               ) == 2, f"Constant {n} is not reused"
    assert len(
        const_names) == 0, f"Not all constants were found: {const_names}"


@pytest.mark.parametrize(
    ("l_type", "r_type"),
    itertools.product(
        [
            float,
            int,
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bfloat16,
            torch.float16,
            torch.float32,
            torch.float64,
        ],
        repeat=2,
    ),
)
@pytest.mark.parametrize("l_scalar", [True, False])
@pytest.mark.parametrize("r_scalar", [True, False])
def test_pytorch_types_promotion(l_type, r_type, l_scalar, r_scalar):
    from openvino.frontend.pytorch.ts_decoder import (TorchScriptPythonDecoder,
                                                      pt_to_ov_type_map)

    class aten_add_t_t(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor):
            return x + y

    class aten_add_int_int(torch.nn.Module):
        def forward(self, x: int, y: int):
            return x + y

    class aten_add_float_float(torch.nn.Module):
        def forward(self, x: float, y: float):
            return x + y

    class aten_add_int_float(torch.nn.Module):
        def forward(self, x: int, y: float):
            return x + y

    class aten_add_float_int(torch.nn.Module):
        def forward(self, x: float, y: int):
            return x + y

    class aten_add_t_int(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: int):
            return x + y

    class aten_add_int_t(torch.nn.Module):
        def forward(self, x: int, y: torch.Tensor):
            return x + y

    class aten_add_t_float(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: float):
            return x + y

    class aten_add_float_t(torch.nn.Module):
        def forward(self, x: float, y: torch.Tensor):
            return x + y

    l_t = "t"
    r_t = "t"

    if isinstance(l_type, type):
        ov_lhs = ops.parameter(PartialShape(
            []), pt_to_ov_type_map.get(l_type.__name__))
        pt_lhs = l_type(5)
        l_t = l_type.__name__
    elif l_scalar:
        ov_lhs = ops.parameter(PartialShape(
            []), pt_to_ov_type_map.get(str(l_type)))
        pt_lhs = torch.tensor(1, dtype=l_type)
    else:
        ov_lhs = ops.parameter(PartialShape(
            [2, 2]), pt_to_ov_type_map.get(str(l_type)))
        pt_lhs = torch.rand([2, 2]).to(dtype=l_type)

    if isinstance(r_type, type):
        ov_rhs = ops.parameter(PartialShape(
            []), pt_to_ov_type_map.get(r_type.__name__))
        pt_rhs = r_type(5)
        r_t = r_type.__name__
    elif r_scalar:
        ov_rhs = ops.parameter(PartialShape(
            []), pt_to_ov_type_map.get(str(r_type)))
        pt_rhs = torch.tensor(1, dtype=r_type)
    else:
        ov_rhs = ops.parameter(PartialShape(
            [2, 2]), pt_to_ov_type_map.get(str(r_type)))
        pt_rhs = torch.rand([2, 2]).to(dtype=r_type)
    model = get_scripted_model(locals().get(f"aten_add_{l_t}_{r_t}")())
    decoder = TorchScriptPythonDecoder(model)
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    lhs_place = im.get_place_by_tensor_name("x.1")
    rhs_place = im.get_place_by_tensor_name("y.1")
    im.set_element_type(lhs_place, ov_lhs.get_output_element_type(0))
    im.set_element_type(rhs_place, ov_rhs.get_output_element_type(0))
    im.set_partial_shape(lhs_place, ov_lhs.get_output_partial_shape(0))
    im.set_partial_shape(rhs_place, ov_rhs.get_output_partial_shape(0))
    om = fe.convert(im)
    pt_out = model(pt_lhs, pt_rhs)
    if isinstance(pt_out, (float, int, bool)):
        pt_out_type = type(pt_out).__name__
        pt_out_shape = []
    else:
        pt_out_type = pt_out.dtype
        pt_out_shape = pt_out.size()
    pt_out_type = pt_to_ov_type_map.get(str(pt_out_type))
    ov_out_type = om.get_output_element_type(0)
    assert pt_out_type == ov_out_type
    print(f"{pt_out_type} == {ov_out_type}")
    assert PartialShape(pt_out_shape) == om.get_output_partial_shape(0)


class ModelTest1(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return {"x1": self.pool(x), "x2": x * 5}


def test_output_dict_names():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    model = ModelTest1()
    decoder = TorchScriptPythonDecoder(
        model, example_input=(torch.randn(1, 3, 224, 224),))
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    om = fe.convert(im)
    assert om.outputs[0].any_name == "x1" and om.outputs[1].any_name == "x2", "Output dict names are not expected"


class ModelTest2(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x), x * 5


def test_output_tuple_names():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

    model = ModelTest2()
    decoder = TorchScriptPythonDecoder(
        model, example_input=(torch.randn(1, 3, 224, 224),))
    fe_manager = FrontEndManager()
    fe = fe_manager.load_by_framework("pytorch")
    im = fe.load(decoder)
    om = fe.convert(im)
    assert len(om.outputs[0].names) == 0 and len(
        om.outputs[1].names) == 0, "Output tuple names must be empty"


def test_patched_16bit_model_converts():
    from openvino.frontend.pytorch import patch_model
    from openvino import convert_model, compile_model
    import copy
    import inspect
    from transformers.pytorch_utils import Conv1D

    class ModelWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(
                torch.nn.Embedding(10, 64),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU()
            )
            self.branch2 = torch.nn.Sequential(
                Conv1D(256, 128),
                torch.nn.Linear(256, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example = (torch.randint(0, 10, [32, 64]), torch.randn(32, 128))
    model_ref = ModelWithLinear()
    with torch.no_grad():
        res_ref = model_ref(*example)
    model_fp16 = copy.deepcopy(model_ref).half()

    patch_model.__make_16bit_traceable(model_fp16)
    # verify torch.nn.Linear signature after patching
    signature = inspect.signature(model_ref.branch1[0].forward).parameters
    assert ["input"] == list(signature)
    # the approach with patching only works for node with no grad
    with torch.no_grad():
        converted_model = convert_model(model_fp16, example_input=example)
    assert converted_model
    cm_fp16 = compile_model(converted_model, "CPU")
    res_fp16 = cm_fp16([x.numpy() for x in example])
    np.testing.assert_allclose(res_fp16[0], res_ref[0].numpy(), atol=1e-2)
    np.testing.assert_allclose(res_fp16[1], res_ref[1].numpy(), atol=1e-2)

    model_bf16 = copy.deepcopy(model_ref).bfloat16()
    patch_model.__make_16bit_traceable(model_bf16)
    # the approach with patching only works for node with no grad
    with torch.no_grad():
        converted_model = convert_model(model_bf16, example_input=example)
    assert converted_model
    cm_bf16 = compile_model(converted_model, "CPU")
    res_bf16 = cm_bf16([x.numpy() for x in example])
    np.testing.assert_allclose(res_bf16[0], res_ref[0].numpy(), atol=1e-2)
    np.testing.assert_allclose(res_bf16[1], res_ref[1].numpy(), atol=1e-2)


def test_patched_16bit_model_with_convert():
    from openvino.frontend.pytorch import patch_model
    from openvino import convert_model, Type

    class ModelWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(32, 64, dtype=torch.float16)
            self.l2 = torch.nn.Linear(64, 32, dtype=torch.float16)

        def forward(self, x):
            x = self.l1(x)
            x = x.to(self.l1.weight.dtype)
            x = self.l2(x)
            return x

    example = (torch.randint(0, 10, [16, 32]),)
    model = ModelWithLinear()
    patch_model.__make_16bit_traceable(model)
    with torch.no_grad():
        converted_model = convert_model(model, example_input=example)
    assert converted_model
    mm_num = 0
    for node in converted_model.get_ordered_ops():
        if node.get_type_name() == "MatMul":
            mm_num += 1
            # verify all matmuls are executed in fp32
            assert node.get_input_element_type(0) == Type.f32
            assert node.get_input_element_type(1) == Type.f32
            assert node.get_output_element_type(0) == Type.f32
    assert mm_num == 2


def test_patched_8bit_model_converts():
    from openvino.frontend.pytorch import patch_model
    from openvino import convert_model, compile_model
    from transformers.pytorch_utils import Conv1D

    class ModelWithLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.branch1 = torch.nn.Sequential(
                torch.nn.Embedding(10, 64),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU()
            )
            self.branch2 = torch.nn.Sequential(
                Conv1D(256, 128),
                torch.nn.Linear(256, 64), torch.nn.ReLU()
            )
            self.buffer = torch.ones(32)

        def forward(self, x1, x2):
            out1 = self.branch1(x1)
            out2 = self.branch2(x2)
            return (out1 + self.buffer, out2)

    example = (torch.randint(0, 10, [32, 64]), torch.randn(32, 128))

    model_ref = ModelWithLinear().to(torch.float8_e4m3fn).float()
    with torch.no_grad():
        res_ref = model_ref(*example)
    model_f8_e4m3 = model_ref.to(torch.float8_e4m3fn)
    patch_model.__make_16bit_traceable(model_f8_e4m3)
    # the approach with patching only works for node with no grad
    with torch.no_grad():
        converted_model = convert_model(model_f8_e4m3, example_input=example)
    assert converted_model
    cm_f8_e4m3 = compile_model(converted_model, "CPU")
    res_f8_e4m3 = cm_f8_e4m3([x.numpy() for x in example])
    np.testing.assert_allclose(res_f8_e4m3[0], res_ref[0].numpy(), atol=1e-2)
    np.testing.assert_allclose(res_f8_e4m3[1], res_ref[1].numpy(), atol=1e-2)

    model_ref = ModelWithLinear().to(torch.float8_e5m2).float()
    with torch.no_grad():
        res_ref = model_ref(*example)
    model_f8_e5m2 = model_ref.to(torch.float8_e5m2)
    patch_model.__make_16bit_traceable(model_f8_e5m2)
    # the approach with patching only works for node with no grad
    with torch.no_grad():
        converted_model = convert_model(model_f8_e5m2, example_input=example)
    assert converted_model
    cm_f8_e5m2 = compile_model(converted_model, "CPU")
    res_f8_e5m2 = cm_f8_e5m2([x.numpy() for x in example])
    np.testing.assert_allclose(res_f8_e5m2[0], res_ref[0].numpy(), atol=1e-2)
    np.testing.assert_allclose(res_f8_e5m2[1], res_ref[1].numpy(), atol=1e-2)


class InlinedInputsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return torch.arange(2048)


def test_inlined_inputs():
    model = InlinedInputsModel()
    model.eval()
    model = torch.compile(model, backend="openvino", options={"testing": 1})
    model()
