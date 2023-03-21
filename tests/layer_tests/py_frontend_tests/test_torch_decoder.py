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
        print(model.inlined_graph)  # will help debugging
        return model

def get_traced_model(model, inputs=[], frozen=True):
    with torch.no_grad():
        model = torch.jit.trace(model, example_inputs=inputs)
        model.eval()
        if frozen:
            model = torch.jit.freeze(model)
        print(model.inlined_graph)  # will help debugging
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


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_fp16_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.float16)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f16
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_fp32_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.float32)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f32
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_fp64_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.float64)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f64
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_bool_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 0], dtype=torch.bool)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.boolean
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_u8_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.uint8)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.u8
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_i8_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.int8)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i8
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_i32_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.int)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i32
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_i64_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.int64)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i64
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_int64_max():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder

    class I64MaxConst(torch.nn.Module):
        def forward(self):
            return 9223372036854775807

    model = get_scripted_model(I64MaxConst())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    int64_const = consts[0]
    print(int64_const)
    nc_decoder = TorchScriptPythonDecoder(model, int64_const)
    assert nc_decoder.as_constant() is not None


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_int_list():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class ListConst(torch.nn.Module):
        def forward(self):
            return [1, 2]

    model = get_scripted_model(ListConst())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    print(some_const)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i32
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_float_list():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class ListConst(torch.nn.Module):
        def forward(self):
            return [float(1), float(2)]

    model = get_scripted_model(ListConst())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    print(some_const)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f32
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_bool_list():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class ListConst(torch.nn.Module):
        def forward(self):
            return [True, False]

    model = get_scripted_model(ListConst())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    print(some_const)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.boolean
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_int_tuple():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class ListConst(torch.nn.Module):
        def forward(self):
            return (1, 2)

    model = get_scripted_model(ListConst())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    print(some_const)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i32
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_float_tuple():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class ListConst(torch.nn.Module):
        def forward(self):
            return (float(1), float(2))

    model = get_scripted_model(ListConst())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    print(some_const)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f32
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_bool_tuple():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class ListConst(torch.nn.Module):
        def forward(self):
            return (True, False)

    model = get_scripted_model(ListConst())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    print(some_const)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.boolean
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_empty_list():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class aten_roll(torch.nn.Module):
        def __init__(self, shifts):
            super(aten_roll, self).__init__()
            self.shits = shifts

        def forward(self, x):
            # roll has optional input dim, which is empty int list by default
            return torch.roll(x, self.shits)

    model = get_scripted_model(aten_roll(1))
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 1
    empty_const = consts[1]
    print(empty_const)
    nc_decoder = TorchScriptPythonDecoder(model, empty_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i32
    assert ov_const[0].get_partial_shape() == PartialShape([0])

@pytest.mark.precommit
def test_pytorch_decoder_can_convert_int_scalar_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value: int = 1

        def forward(self):
            # Reproduce specific case where prim::Constant for `self.value + 1`
            # would create torch.Node with output being Tensor with IValue  of type int.
            return torch.add(torch.tensor([1], dtype=torch.int32), self.value + 1)

    model = get_traced_model(SomeTensor(), frozen=False)
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[6]
    node_output = list(some_const.outputs())[0]
    assert node_output.isCompleteTensor()
    assert isinstance(node_output.toIValue(), int)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i32
    assert ov_const[0].get_partial_shape() == PartialShape([])

@pytest.mark.precommit
def test_pytorch_decoder_can_convert_float_scalar_tensor():
    from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    from openvino.runtime import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value: float = 1.

        def forward(self):
            # Reproduce specific case where prim::Constant for `self.value + 1`
            # would create nore with output being Tensor with IValue  of type float.
            return torch.add(torch.tensor([1.], dtype=torch.float), self.value + 1)


    model = get_traced_model(SomeTensor(), frozen=False)
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
            "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[6]
    node_output = list(some_const.outputs())[0]
    assert node_output.isCompleteTensor()
    assert isinstance(node_output.toIValue(), float)
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f32
    assert ov_const[0].get_partial_shape() == PartialShape([])
