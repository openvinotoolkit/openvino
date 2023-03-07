# -*- coding: utf-8 -*-
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend import FrontEndManager, ConversionExtension
from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
from openvino.frontend import OpExtension
from openvino.frontend import NodeContext

import openvino.runtime.opset10 as ops
from openvino.runtime import shutdown
from openvino.runtime import Core

import pytest
import numpy as np
import torch

from tests.test_frontend.common import get_builtin_extensions_path


fem = FrontEndManager()


def skip_if_pytorch_frontend_is_disabled():
    front_ends = fem.get_available_front_ends()
    if "pytorch" not in front_ends:
        pytest.skip()


def teardown_module():
    shutdown()


def test_conversion_extension():
    skip_if_pytorch_frontend_is_disabled()

    class Elu(torch.nn.Module):
        def __init__(self, alpha):
            super(Elu, self).__init__()
            self.alpha = alpha

        def forward(self, inp):
            return torch.nn.functional.elu(inp, self.alpha)

    model = Elu(alpha=0.1)
    model.eval()
    model = torch.jit.script(model)
    model = torch.jit.freeze(model)
    decoder = TorchScriptPythonDecoder(model)

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


def test_op_extension():
    skip_if_pytorch_frontend_is_disabled()

    class Elu(torch.nn.Module):
        def __init__(self, alpha):
            super(Elu, self).__init__()
            self.alpha = alpha

        def forward(self, inp):
            return torch.nn.functional.elu(inp, self.alpha)

    model = Elu(alpha=0.1)
    model.eval()
    model = torch.jit.script(model)
    model = torch.jit.freeze(model)
    decoder = TorchScriptPythonDecoder(model)

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
