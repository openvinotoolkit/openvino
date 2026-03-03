# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import itertools
import math
import os
import re
import sys
import logging
import platform
from pathlib import Path

import torch
import numpy as np
import pytest

from openvino.frontend import FrontEndManager, ConversionExtension, NodeContext
from openvino import PartialShape, Type
import openvino.opset10 as ops
import openvino.properties.hint as hints

logging.basicConfig(level=logging.DEBUG)

orig_compile = torch.compile
torch.compile = lambda func: func
default_cfg = {hints.inference_precision: Type.f32}


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
        "Parameter", "Elu", "Constant", "ConvertLike", "Multiply", "Result"]

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


def verify_model(model, example_input, expected_ops):
    import numpy as np
    import openvino as ov
    # Convert and compile the model
    converted_model = ov.convert_model(model, example_input=(example_input,))
    assert converted_model, "Model conversion failed."
    compiled_model = ov.compile_model(converted_model, "CPU", default_cfg)
    assert compiled_model, "Model compilation failed."

    # Verify model operations
    actual_ops = [n.get_type_name() for n in converted_model.get_ordered_ops()]
    assert actual_ops == expected_ops, f"Expected {expected_ops}, but got {actual_ops}."

    # Test model execution
    test_input = example_input.numpy(force=True)
    res = compiled_model((test_input,))
    ref = model(torch.from_numpy(test_input))
    rtol, atol = 1e-7, 0
    if platform.machine() in ('arm', 'armv7l', 'aarch64', 'arm64', 'ARM64'):
        rtol, atol = 0.1, 0.001
    if isinstance(ref, tuple):
        for i, ref_part in enumerate(ref):
            np.testing.assert_allclose(res[i], ref_part.numpy(), rtol, atol)
    else:
        np.testing.assert_allclose(res[0], ref.numpy(), rtol, atol)


def test_inlined_extension():
    from openvino.frontend.pytorch import inlined_extension
    rng = np.random.default_rng(42)

    @inlined_extension
    def numpy_cos(x):
        return torch.from_numpy(np.cos(x.numpy(force=True)))

    class ModelWithModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu_module = torch.nn.ReLU()

        def forward(self, x):
            x = x.to(torch.float32)
            return numpy_cos(x) + self.relu_module(x)

    model = ModelWithModule()
    example = torch.from_numpy(rng.random([100], dtype=np.float32))
    expected_ops = ["Parameter", "InlinedCustomOp", "Relu", "Add", "Result"]
    verify_model(model, example, expected_ops)


def test_multiple_inlined_extension():
    from openvino.frontend.pytorch import inlined_extension
    rng = np.random.default_rng(42)

    @inlined_extension
    def numpy_roll(x):
        return torch.from_numpy(np.roll(x.numpy(force=True), 10, 0))

    @inlined_extension
    def numpy_cos(x):
        return torch.from_numpy(np.cos(x.numpy(force=True)))

    class ModelWithModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu_module = torch.nn.ReLU()

        def forward(self, x):
            x = x.to(torch.float32)
            x = numpy_roll(x) + self.relu_module(x)
            x = numpy_cos(x) + self.relu_module(x)
            return numpy_roll(x) + self.relu_module(x)

    model = ModelWithModule()
    example = torch.from_numpy(rng.random([100], dtype=np.float32))
    expected_ops = ["Parameter", "InlinedCustomOp", "Relu", "Add", "InlinedCustomOp", "Relu", "Add", "InlinedCustomOp", "Relu", "Add", "Result"]
    verify_model(model, example, expected_ops)


def test_inlined_extension_multiple_outputs():
    from openvino.frontend.pytorch import inlined_extension
    rng = np.random.default_rng(42)

    @inlined_extension
    def numpy_split(x):
        np_array = x.numpy(force=True)
        midpoint = np_array.shape[0] // 2
        part1 = np_array[:midpoint]
        part2 = np_array[midpoint:]
        return torch.from_numpy(part1), torch.from_numpy(part2)

    class ModelWithMultipleOutputs(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu_module = torch.nn.ReLU()

        def forward(self, x):
            x = x.to(torch.float32)
            part1, part2 = numpy_split(x)
            return self.relu_module(part1), self.relu_module(part2)

    model = ModelWithMultipleOutputs()
    example = torch.from_numpy(rng.random([100], dtype=np.float32))
    expected_ops = ["Parameter", "InlinedCustomOp", "Relu", "Result", "Relu", "Result"]
    verify_model(model, example, expected_ops)


def test_inlined_extension_with_torch_model():
    from openvino.frontend.pytorch import inlined_extension
    rng = np.random.default_rng(42)

    # Define a simple PyTorch model
    class SimpleTorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(100, 100)

        def forward(self, x):
            return self.linear(x)

    # Instantiate the model
    simple_model = SimpleTorchModel()

    @inlined_extension
    def model_based_extension(x):
        return simple_model(x)

    class ModelWithInlinedTorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu_module = torch.nn.ReLU()

        def forward(self, x):
            x = x.to(torch.float32)
            return model_based_extension(x) + self.relu_module(x)

    model = ModelWithInlinedTorchModel()
    example = torch.from_numpy(rng.random([100], dtype=np.float32))
    expected_ops = ["Parameter", "InlinedCustomOp", "Relu", "Add", "Result"]
    verify_model(model, example, expected_ops)


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
    try:
        # verify torch.nn.Linear signature after patching
        signature = inspect.signature(model_ref.branch1[0].forward).parameters
        assert ["input"] == list(signature)
        # the approach with patching only works for node with no grad
        with torch.no_grad():
            converted_model = convert_model(model_fp16, example_input=example)
        assert converted_model
        cm_fp16 = compile_model(converted_model, "CPU", default_cfg)
        res_fp16 = cm_fp16([x.numpy() for x in example])
        np.testing.assert_allclose(res_fp16[0], res_ref[0].numpy(), atol=1e-2)
        np.testing.assert_allclose(res_fp16[1], res_ref[1].numpy(), atol=1e-2)
    finally:
        patch_model._unpatch_torch_functions()

    model_bf16 = copy.deepcopy(model_ref).bfloat16()
    patch_model.__make_16bit_traceable(model_bf16)
    try:
        # the approach with patching only works for node with no grad
        with torch.no_grad():
            converted_model = convert_model(model_bf16, example_input=example)
        assert converted_model
        cm_bf16 = compile_model(converted_model, "CPU", default_cfg)
        res_bf16 = cm_bf16([x.numpy() for x in example])
        np.testing.assert_allclose(res_bf16[0], res_ref[0].numpy(), atol=1e-2)
        np.testing.assert_allclose(res_bf16[1], res_ref[1].numpy(), atol=1e-2)
    finally:
        patch_model._unpatch_torch_functions()


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
    try:
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
    finally:
        patch_model._unpatch_torch_functions()


def test_patched_8bit_model_converts_e4m3fn():
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
    try:
        # the approach with patching only works for node with no grad
        with torch.no_grad():
            converted_model = convert_model(model_f8_e4m3, example_input=example)
        assert converted_model
        cm_f8_e4m3 = compile_model(converted_model, "CPU", default_cfg)
        res_f8_e4m3 = cm_f8_e4m3([x.numpy() for x in example])
        np.testing.assert_allclose(res_f8_e4m3[0], res_ref[0].numpy(), atol=1e-2)
        np.testing.assert_allclose(res_f8_e4m3[1], res_ref[1].numpy(), atol=1e-2)
    finally:
        patch_model._unpatch_torch_functions()


@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64'],
    reason="PyTorch float8_e5m2 cleanup deadlock on macOS ARM64. Ticket: 172658"
)
def test_patched_8bit_model_converts_e5m2():
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

    model_ref = ModelWithLinear().to(torch.float8_e5m2).float()
    with torch.no_grad():
        res_ref = model_ref(*example)
    model_f8_e5m2 = model_ref.to(torch.float8_e5m2)
    patch_model.__make_16bit_traceable(model_f8_e5m2)
    try:
        # the approach with patching only works for node with no grad
        with torch.no_grad():
            converted_model = convert_model(model_f8_e5m2, example_input=example)
        assert converted_model
        cm_f8_e5m2 = compile_model(converted_model, "CPU", default_cfg)
        res_f8_e5m2 = cm_f8_e5m2([x.numpy() for x in example])
        np.testing.assert_allclose(res_f8_e5m2[0], res_ref[0].numpy(), atol=1e-2)
        np.testing.assert_allclose(res_f8_e5m2[1], res_ref[1].numpy(), atol=1e-2)
    finally:
        patch_model._unpatch_torch_functions()


def test_patched_16bit_model_with_bmm():
    from openvino.frontend.pytorch import patch_model
    from openvino import convert_model, compile_model
    import copy

    rng = torch.Generator().manual_seed(42)

    class MoEStyleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.expert_weights = torch.nn.Parameter(
                torch.randint(-4, 5, (4, 32, 64), generator=rng).float() * 0.25)
            self.linear = torch.nn.Linear(64, 32, bias=False)
            self.linear.weight.data = torch.randint(-4, 5, (32, 64), generator=rng).float() * 0.25

        def forward(self, x):
            batch, seq, _ = x.shape
            x_expanded = x.reshape(1, batch * seq, -1).expand(4, -1, -1)
            expert_out = torch.bmm(x_expanded, self.expert_weights)
            out = expert_out.mean(dim=0).reshape(batch, seq, -1)
            return self.linear(out)

    example = (torch.randint(-4, 5, (2, 8, 32), generator=rng).float() * 0.25,)
    model_ref = MoEStyleModel()
    with torch.no_grad():
        res_ref = model_ref(*example)

    try:
        model_bf16 = copy.deepcopy(model_ref).bfloat16()
        patch_model.__make_16bit_traceable(model_bf16)
        with torch.no_grad():
            converted_model = convert_model(model_bf16, example_input=example)
        assert converted_model
        cm_bf16 = compile_model(converted_model, "CPU")
        res_bf16 = cm_bf16([x.numpy() for x in example])
        np.testing.assert_allclose(res_bf16[0], res_ref.numpy(), atol=1e-2)

        model_fp16 = copy.deepcopy(model_ref).half()
        patch_model.__make_16bit_traceable(model_fp16)
        with torch.no_grad():
            converted_model = convert_model(model_fp16, example_input=example)
        assert converted_model
        cm_fp16 = compile_model(converted_model, "CPU")
        res_fp16 = cm_fp16([x.numpy() for x in example])
        np.testing.assert_allclose(res_fp16[0], res_ref.numpy(), atol=1e-2)
    finally:
        patch_model._unpatch_torch_functions()


@pytest.mark.skipif(sys.platform.lower().startswith("win"), reason="CVS-174725")
def test_patched_bitnet_model_converts():
    from openvino import convert_model, compile_model
    from transformers.integrations.bitnet import AutoBitLinear, pack_weights
    from transformers import PretrainedConfig, BitNetQuantConfig

    rng = torch.Generator().manual_seed(42)

    class TestModel(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.config = PretrainedConfig(quantization_config=BitNetQuantConfig(linear_class="autobitlinear"))
            self.linear = AutoBitLinear(size[0], size[1], bias=True, use_rms_norm=True)
            w = torch.randint(-1, 2, (size[1], size[0]), dtype=torch.float32, generator=rng)
            self.linear.weight = torch.nn.Parameter(w)
            self.linear.original_weight = pack_weights(self.linear.weight.data.clone())

        def forward(self, x):
            return self.linear(x)

    size = (32, 64)
    x = torch.randn(1, size[0], generator=rng)
    model = TestModel(size)
    with torch.no_grad():
        res_ref = model(x)

    with torch.no_grad():
        converted_model = convert_model(model, example_input=(torch.randn(1, size[0], generator=rng),))
    assert converted_model
    cm = compile_model(converted_model, "CPU", default_cfg)
    res = cm([x.numpy()])
    rtol, atol = 1e-4, 1e-4
    if platform.machine() in ('arm', 'armv7l', 'aarch64', 'arm64', 'ARM64'):
        rtol, atol = 0.5, 0.1
    np.testing.assert_allclose(res[0], res_ref.numpy(), rtol=rtol, atol=atol)


class InlinedInputsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return torch.arange(2048)


def test_inlined_inputs():
    model = InlinedInputsModel()
    model.eval()
    model = orig_compile(model, backend="openvino", options={"testing": 1})
    model()


# --- dynamo (torch.export) tests ---

def _torch_version_at_least(version_str):
    from packaging.version import parse
    return parse(torch.__version__.split('+')[0].split('dev')[0]) >= parse(version_str)


@pytest.mark.skipif(not _torch_version_at_least("2.6"),
                    reason="Dim.AUTO requires PyTorch >= 2.6")
class TestBuildDynamicShapes:
    """Unit tests for _build_dynamic_shapes covering many input combinations."""

    @staticmethod
    def _make_spec(shape):
        """Create an _InputCutInfo with the given PartialShape."""
        from openvino.tools.ovc.cli_parser import _InputCutInfo
        return _InputCutInfo(name=None, shape=PartialShape(shape))

    # ── input_specs=None → always None ──────────────────────────────

    def test_no_specs_returns_none_single_input(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        example_input = (torch.randn(2, 3),)
        result = _build_dynamic_shapes(example_input)
        assert result is None

    def test_no_specs_returns_none_multi_input(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        example_input = (torch.randn(4,), torch.randn(4,))
        result = _build_dynamic_shapes(example_input)
        assert result is None

    def test_no_specs_returns_none_dict_input(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        example_input = {"x": torch.randn(2, 3), "y": torch.randn(2, 3)}
        result = _build_dynamic_shapes(example_input)
        assert result is None

    def test_no_specs_returns_none_single_tensor(self):
        """Bare tensor (not wrapped in tuple) with no specs."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        example_input = torch.randn(2, 3, 4)
        result = _build_dynamic_shapes(example_input)
        assert result is None

    # ── All dims dynamic (-1) ───────────────────────────────────────

    def test_all_dims_dynamic_1d(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(5),)
        inp = [self._make_spec([-1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert isinstance(result, tuple)
        assert result[0] == {0: Dim.AUTO}

    def test_all_dims_dynamic_2d(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 3),)
        inp = [self._make_spec([-1, -1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO, 1: Dim.AUTO}

    def test_all_dims_dynamic_4d(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(1, 3, 224, 224),)
        inp = [self._make_spec([-1, -1, -1, -1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO, 1: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}

    # ── Only batch dynamic ──────────────────────────────────────────

    def test_batch_only_dynamic_4d(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 3, 32, 32),)
        inp = [self._make_spec([-1, 3, 32, 32])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}

    def test_batch_only_dynamic_3d(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(4, 10, 512),)
        inp = [self._make_spec([-1, 10, 512])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}

    # ── Mixed dynamic / static dims ─────────────────────────────────

    def test_batch_and_seqlen_dynamic(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(1, 128, 768),)
        inp = [self._make_spec([-1, -1, 768])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO, 1: Dim.AUTO}

    def test_only_spatial_dynamic(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(1, 3, 64, 64),)
        inp = [self._make_spec([1, 3, -1, -1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {2: Dim.AUTO, 3: Dim.AUTO}

    def test_only_middle_dim_dynamic(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(4, 10, 16),)
        inp = [self._make_spec([4, -1, 16])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {1: Dim.AUTO}

    # ── All static dims → None per input ────────────────────────────

    def test_all_static_returns_none_for_input(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        example_input = (torch.randn(2, 3, 4),)
        inp = [self._make_spec([2, 3, 4])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result is not None  # tuple is returned (not None), but element is None
        assert result[0] is None

    # ── Multiple inputs ─────────────────────────────────────────────

    def test_two_inputs_both_dynamic(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 8), torch.randn(2, 8))
        inp = [self._make_spec([-1, 8]), self._make_spec([-1, 8])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}
        assert result[1] == {0: Dim.AUTO}

    def test_two_inputs_one_dynamic_one_static(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 3, 32, 32), torch.randn(2, 10))
        inp = [self._make_spec([-1, 3, 32, 32]), self._make_spec([2, 10])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}
        assert result[1] is None

    def test_three_inputs_mixed(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(1, 128, 64), torch.randn(1, 128, 64), torch.randn(1, 128, 64))
        inp = [self._make_spec([-1, -1, 64])] * 3
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        for i in range(3):
            assert result[i] == {0: Dim.AUTO, 1: Dim.AUTO}

    # ── Dict inputs ─────────────────────────────────────────────────

    def test_dict_input_single_dynamic(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = {"x": torch.randn(4, 16), "y": torch.randn(4, 16)}
        inp = [self._make_spec([-1, 16]), self._make_spec([-1, 16])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result["x"] == {0: Dim.AUTO}
        assert result["y"] == {0: Dim.AUTO}

    def test_dict_input_mixed_shapes(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = {"image": torch.randn(1, 3, 64, 64), "mask": torch.randn(1, 1, 64, 64)}
        inp = [self._make_spec([-1, 3, -1, -1]), self._make_spec([-1, 1, -1, -1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result["image"] == {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}
        assert result["mask"] == {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}

    def test_dict_input_all_static(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        example_input = {"a": torch.randn(2, 3), "b": torch.randn(2, 3)}
        inp = [self._make_spec([2, 3]), self._make_spec([2, 3])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result["a"] is None
        assert result["b"] is None

    # ── List inputs (auto-converted to tuple) ───────────────────────

    def test_list_input(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = [torch.randn(3, 5)]
        inp = [self._make_spec([-1, 5])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}

    # ── Bare single tensor input ────────────────────────────────────

    def test_bare_tensor_input(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = torch.randn(2, 10)
        inp = [self._make_spec([-1, 10])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}

    # ── Scalar tensor → None ────────────────────────────────────────

    def test_scalar_tensor_returns_none(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        example_input = (torch.tensor(3.14),)
        inp = [self._make_spec([])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] is None

    # ── Spec with None shape ────────────────────────────────────────

    def test_spec_with_none_shape(self):
        """Spec that has shape=None should not produce dynamic dims."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino.tools.ovc.cli_parser import _InputCutInfo
        example_input = (torch.randn(2, 3),)
        inp = [_InputCutInfo(name=None, shape=None)]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] is None

    # ── More specs than inputs (extra specs ignored) ────────────────

    def test_extra_specs_ignored(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 4),)
        inp = [self._make_spec([-1, 4]), self._make_spec([-1, 8])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}

    # ── Fewer specs than inputs (missing specs → None) ──────────────

    def test_fewer_specs_than_inputs(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 4), torch.randn(2, 4))
        inp = [self._make_spec([-1, 4])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}
        assert result[1] is None  # no spec → static

    # ── High-dimensional tensors ────────────────────────────────────

    def test_5d_tensor(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 3, 16, 224, 224),)
        inp = [self._make_spec([-1, 3, -1, 224, 224])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO, 2: Dim.AUTO}

    def test_6d_tensor_all_dynamic(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(1, 2, 3, 4, 5, 6),)
        inp = [self._make_spec([-1, -1, -1, -1, -1, -1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO, 1: Dim.AUTO, 2: Dim.AUTO,
                              3: Dim.AUTO, 4: Dim.AUTO, 5: Dim.AUTO}

    # ── Multiple inputs with different ranks ────────────────────────

    def test_inputs_different_ranks(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(8), torch.randn(4, 8), torch.randn(2, 4, 8))
        inp = [self._make_spec([-1]), self._make_spec([-1, -1]), self._make_spec([-1, -1, -1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] == {0: Dim.AUTO}
        assert result[1] == {0: Dim.AUTO, 1: Dim.AUTO}
        assert result[2] == {0: Dim.AUTO, 1: Dim.AUTO, 2: Dim.AUTO}

    # ── Spec shape shorter than tensor rank (extra dims stay static)

    def test_spec_shorter_than_tensor(self):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        example_input = (torch.randn(2, 3, 32, 32),)
        inp = [self._make_spec([-1, 3])]  # only covers first 2 of 4 dims
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        # Only dim 0 is -1 in spec, dim 1 is 3 (static), dims 2,3 not in spec → static
        assert result[0] == {0: Dim.AUTO}

    # ── Dimension constraints (min/max bounds) ──────────────────────

    def test_constrained_batch_dim(self):
        """Dimension(1, 10) → Dim with min=1, max=10."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        from torch.export import Dim
        example_input = (torch.randn(2, 3, 32, 32),)
        inp = [self._make_spec([Dimension(1, 10), 3, 32, 32])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        d = result[0][0]
        assert d is not Dim.AUTO
        assert d.min == 1
        assert d.max == 10

    def test_constrained_lower_bound_only(self):
        """Dimension(1, -1) → Dim with min=1, unbounded max."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        from torch.export import Dim
        example_input = (torch.randn(4, 16),)
        inp = [self._make_spec([Dimension(1, -1), 16])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        d = result[0][0]
        assert d is not Dim.AUTO
        assert d.min == 1

    def test_constrained_upper_bound_only(self):
        """Dimension(0, 100) → Dim with max=100."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        from torch.export import Dim
        example_input = (torch.randn(4, 16),)
        inp = [self._make_spec([Dimension(0, 100), 16])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        d = result[0][0]
        assert d is not Dim.AUTO
        assert d.max == 100

    def test_constrained_mixed_with_fully_dynamic(self):
        """Mix constrained dims, fully dynamic dims, and static dims."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        from torch.export import Dim
        example_input = (torch.randn(1, 3, 64, 64),)
        inp = [self._make_spec([Dimension(1, 8), 3, -1, -1])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        # dim 0: constrained
        assert result[0][0] is not Dim.AUTO
        assert result[0][0].min == 1
        assert result[0][0].max == 8
        # dims 2,3: fully dynamic
        assert result[0][2] is Dim.AUTO
        assert result[0][3] is Dim.AUTO
        # dim 1 (static 3) not present
        assert 1 not in result[0]

    def test_constrained_all_dims(self):
        """All dims have constraints."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        example_input = (torch.randn(2, 8),)
        inp = [self._make_spec([Dimension(1, 4), Dimension(4, 16)])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0][0].min == 1
        assert result[0][0].max == 4
        assert result[0][1].min == 4
        assert result[0][1].max == 16

    def test_constrained_multiple_inputs(self):
        """Two inputs, each with different constraints."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        from torch.export import Dim
        example_input = (torch.randn(2, 8), torch.randn(2, 16))
        inp = [self._make_spec([Dimension(1, 4), 8]),
               self._make_spec([-1, Dimension(8, 32)])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        # Input 0: dim 0 constrained
        assert result[0][0].min == 1
        assert result[0][0].max == 4
        # Input 1: dim 0 fully dynamic, dim 1 constrained
        assert result[1][0] is Dim.AUTO
        assert result[1][1].min == 8
        assert result[1][1].max == 32

    def test_constrained_dict_input(self):
        """Dict input with constrained dimensions."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        from torch.export import Dim
        example_input = {"x": torch.randn(2, 3), "y": torch.randn(2, 3)}
        inp = [self._make_spec([Dimension(1, 8), 3]),
               self._make_spec([-1, 3])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        # "x": dim 0 constrained
        assert result["x"][0].min == 1
        assert result["x"][0].max == 8
        # "y": dim 0 fully dynamic
        assert result["y"][0] is Dim.AUTO

    def test_constrained_all_static_still_none(self):
        """Static Dimension values should be treated same as plain ints."""
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from openvino import Dimension
        example_input = (torch.randn(2, 3),)
        inp = [self._make_spec([Dimension(2), Dimension(3)])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        assert result[0] is None

    # ── Parametrized: many rank / dynamic-dim combos ────────────────

    @pytest.mark.parametrize("ndim,dyn_dims", [
        (1, [0]),
        (2, [0]),
        (2, [1]),
        (2, [0, 1]),
        (3, [0]),
        (3, [2]),
        (3, [0, 2]),
        (4, [0]),
        (4, [0, 2, 3]),
        (4, [0, 1, 2, 3]),
        (5, [0, 2]),
    ])
    def test_parametrized_dynamic_dims(self, ndim, dyn_dims):
        from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import _build_dynamic_shapes
        from torch.export import Dim
        tensor_shape = [(d + 2) for d in range(ndim)]
        example_input = (torch.randn(*tensor_shape),)
        inp = [self._make_spec([(-1 if d in dyn_dims else (d + 2)) for d in range(ndim)])]
        result = _build_dynamic_shapes(example_input, input_specs=inp)
        if dyn_dims:
            assert result[0] == {d: Dim.AUTO for d in dyn_dims}
        else:
            assert result[0] is None


class TestConvertModelDynamo:
    """Tests for convert_model with dynamo=True (torch.export path)."""

    def test_basic(self):
        from openvino import convert_model, compile_model

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 32)

            def forward(self, x):
                return torch.nn.functional.relu(self.linear(x))

        model = SimpleModel()
        example = (torch.randn(2, 16),)

        with torch.no_grad():
            ref = model(*example)

        ov_model = convert_model(model, example_input=example, dynamo=True)
        assert ov_model is not None

        cm = compile_model(ov_model, "CPU", default_cfg)
        res = cm([e.numpy() for e in example])
        np.testing.assert_allclose(res[0], ref.numpy(), atol=1e-4, rtol=1e-4)

    def test_multi_input(self):
        from openvino import convert_model, compile_model

        class MultiInputModel(torch.nn.Module):
            def forward(self, x, y):
                return x + y, x * y

        model = MultiInputModel()
        example = (torch.randn(2, 8), torch.randn(2, 8))

        with torch.no_grad():
            ref = model(*example)

        ov_model = convert_model(model, example_input=example, dynamo=True)
        assert ov_model is not None

        cm = compile_model(ov_model, "CPU", default_cfg)
        res = cm([e.numpy() for e in example])
        np.testing.assert_allclose(res[0], ref[0].numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(res[1], ref[1].numpy(), atol=1e-5, rtol=1e-5)

    def test_dynamic_shapes(self):
        from openvino import convert_model, compile_model

        class DynModel(torch.nn.Module):
            def forward(self, x):
                return x * 2.0

        model = DynModel()
        example = (torch.randn(2, 4),)

        # No 'input' provided → fully static export
        ov_model = convert_model(model, example_input=example, dynamo=True)
        assert ov_model is not None

        input_shape = ov_model.input(0).get_partial_shape()
        assert input_shape.is_static, (
            f"Expected static shape without 'input', got {input_shape}")

        # With 'input' specifying dynamic dims → dynamic export
        ov_model_dyn = convert_model(
            model, example_input=example, dynamo=True,
            input=(PartialShape([-1, -1]),))
        assert ov_model_dyn is not None

        input_shape_dyn = ov_model_dyn.input(0).get_partial_shape()
        assert input_shape_dyn.is_dynamic, (
            f"Expected dynamic shape with 'input', got {input_shape_dyn}")

        # Inference with different shapes should work
        cm = compile_model(ov_model_dyn, "CPU", default_cfg)
        for shape in [(1, 4), (3, 4), (5, 8)]:
            inp = np.random.randn(*shape).astype(np.float32)
            res = cm([inp])
            np.testing.assert_allclose(res[0], inp * 2.0, atol=1e-5, rtol=1e-5)

    def test_no_example_input_raises(self):
        from openvino import convert_model

        class Dummy(torch.nn.Module):
            def forward(self, x):
                return x

        with pytest.raises(Exception, match="example_input is required when dynamo=True"):
            convert_model(Dummy(), dynamo=True)

    def test_dict_example_input(self):
        from openvino import convert_model, compile_model

        class DictModel(torch.nn.Module):
            def forward(self, a, b):
                return a - b

        model = DictModel()
        example = {"a": torch.randn(3, 5), "b": torch.randn(3, 5)}

        with torch.no_grad():
            ref = model(**example)

        ov_model = convert_model(model, example_input=example, dynamo=True)
        assert ov_model is not None

        cm = compile_model(ov_model, "CPU", default_cfg)
        res = cm([v.numpy() for v in example.values()])
        np.testing.assert_allclose(res[0], ref.numpy(), atol=1e-5, rtol=1e-5)

    def test_input_shapes_constrain_dynamism(self):
        """When 'input' specifies shapes, only -1 dims become dynamic."""
        from openvino import convert_model, compile_model

        class ConvModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        model = ConvModel()
        example = (torch.randn(1, 3, 32, 32),)

        # Only batch dim is dynamic (-1), spatial dims are fixed
        ov_model = convert_model(
            model, example_input=example, dynamo=True,
            input=(PartialShape([-1, 3, 32, 32]),))
        assert ov_model is not None

        ps = ov_model.input(0).get_partial_shape()
        # Batch dim should be dynamic
        assert ps[0].is_dynamic, f"Expected dynamic batch dim, got {ps}"
        # Channel and spatial dims should be static
        assert ps[1].get_length() == 3
        assert ps[2].get_length() == 32
        assert ps[3].get_length() == 32

        # Inference with different batch sizes should work
        cm = compile_model(ov_model, "CPU", default_cfg)
        for batch in [1, 2, 4]:
            inp = np.random.randn(batch, 3, 32, 32).astype(np.float32)
            res = cm([inp])
            assert res[0].shape[0] == batch
