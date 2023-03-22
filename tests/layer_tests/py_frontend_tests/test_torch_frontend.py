# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from openvino.frontend import FrontEndManager, ConversionExtension, NodeContext, OpExtension
from openvino.runtime import PartialShape, Type
import openvino.runtime.opset10 as ops

from pathlib import Path
import glob
import re
import os


class aten_relu(torch.nn.Module):
    def forward(self, x):
        return x, torch.nn.functional.relu(x)


def get_scripted_model(model):
    with torch.no_grad():
        model = torch.jit.script(model)
        model.eval()
        model = torch.jit.freeze(model)
        print(model.inlined_graph)  # will help debugging
        return model


def test_pytorch_fe_set_input_shape():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

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
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

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
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

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


def test_conversion_extension():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

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
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "Elu", "Result"]

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

    fe.add_extension(ConversionExtension("aten::elu", convert_elu))
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "Constant", "Greater", "Exp",
                                                                              "Constant", "Subtract", "Constant",
                                                                              "Multiply", "Select", "Result"]


def get_builtin_extensions_path():
    base_path = Path(__file__).parent.parent.parent.parent
    paths = glob.glob(os.path.join(base_path, "bin", "*", "*", "*test_builtin_extensions*"))
    for path in paths:
        if re.search(r"(lib)?test_builtin_extensions.?\.(dll|so)", path):
            return path
    raise RuntimeError("Unable to find test_builtin_extensions")


def test_op_extension():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

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
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "Elu", "Result"]

    fe.add_extension(get_builtin_extensions_path())
    converted_model = fe.convert(input_model)
    assert converted_model
    assert [n.get_type_name() for n in converted_model.get_ordered_ops()] == ["Parameter", "CustomElu", "Result"]
