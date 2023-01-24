# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest


class AtenDiv(torch.nn.Module):
    # aten::div can have str or NoneType constant
    def __init__(self, rounding_mode):
        super(AtenDiv, self).__init__()
        self.rounding_mode = rounding_mode

    def forward(self, input_tensor, other_tensor):
        return torch.div(input_tensor, other_tensor, rounding_mode=self.rounding_mode)


def get_scripted_model(model):
    with torch.no_grad():
        model = torch.jit.script(model)
        model.eval()
        model = torch.jit.freeze(model)
        return model


@pytest.mark.precommit
def test_pytorch_decoder_get_output_type_str():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType

    model = get_scripted_model(AtenDiv("trunc"))
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    # div model has exactly 1 constant
    assert len(consts) > 0
    str_const = consts[0]
    assert isinstance(list(str_const.outputs())[0].type(), torch.StringType)
    nc_decoder = TorchScriptPythonDecoder(model, str_const)
    assert isinstance(nc_decoder.get_output_type(0).value, DecoderType.Str)


@pytest.mark.precommit
def test_pytorch_decoder_get_output_type_none():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType

    model = get_scripted_model(AtenDiv(None))
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    # div model has exactly 1 constant
    assert len(consts) > 0
    none_const = consts[0]
    assert isinstance(list(none_const.outputs())[0].type(), torch.NoneType)
    nc_decoder = TorchScriptPythonDecoder(model, none_const)
    assert isinstance(nc_decoder.get_output_type(0).value, DecoderType.PyNone)


@pytest.mark.precommit
def test_pytorch_decoder_get_input_type_str():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType

    model = get_scripted_model(AtenDiv("trunc"))
    divs = [n for n in model.inlined_graph.nodes() if n.kind() == "aten::div"]
    assert len(divs) > 0
    div_node = divs[0]
    assert isinstance(list(div_node.inputs())[2].type(), torch.StringType)
    nc_decoder = TorchScriptPythonDecoder(model, div_node)
    assert isinstance(nc_decoder.get_input_type(2).value, DecoderType.Str)


@pytest.mark.precommit
def test_pytorch_decoder_get_input_type_none():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType

    model = get_scripted_model(AtenDiv(None))
    divs = [n for n in model.inlined_graph.nodes() if n.kind() == "aten::div"]
    assert len(divs) > 0
    div_node = divs[0]
    assert isinstance(list(div_node.inputs())[2].type(), torch.NoneType)
    nc_decoder = TorchScriptPythonDecoder(model, div_node)
    assert isinstance(nc_decoder.get_input_type(2).value, DecoderType.PyNone)
