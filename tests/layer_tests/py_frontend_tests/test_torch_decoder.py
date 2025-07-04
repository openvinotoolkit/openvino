# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from packaging import version


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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType

    model = get_scripted_model(AtenDiv(None))
    divs = [n for n in model.inlined_graph.nodes() if n.kind() == "aten::div"]
    assert len(divs) > 0
    div_node = divs[0]
    assert isinstance(list(div_node.inputs())[2].type(), torch.NoneType)
    nc_decoder = TorchScriptPythonDecoder(model, div_node)
    assert isinstance(nc_decoder.get_input_type(2).value, DecoderType.PyNone)


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_f8_e4m3_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.float8_e4m3fn)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f8e4m3
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_f8_e5m2_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.float8_e5m2)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.f8e5m2
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_fp16_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
def test_pytorch_decoder_can_convert_bf16_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.bfloat16)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.bf16
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_fp32_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
def test_pytorch_decoder_can_convert_complex64_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            r = torch.tensor([1, 2], dtype=torch.float32)
            i = torch.tensor([3, 4], dtype=torch.float32)
            return torch.complex(r, i)

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
    assert ov_const[0].get_partial_shape() == PartialShape([2, 2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_complex128_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            r = torch.tensor([1, 2], dtype=torch.float64)
            i = torch.tensor([3, 4], dtype=torch.float64)
            return torch.complex(r, i)

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
    assert ov_const[0].get_partial_shape() == PartialShape([2, 2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_bool_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
def test_pytorch_decoder_can_convert_i16_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

    class SomeTensor(torch.nn.Module):
        def forward(self):
            return torch.tensor([1, 2], dtype=torch.int16)

    model = get_scripted_model(SomeTensor())
    consts = [n for n in model.inlined_graph.nodes() if n.kind() ==
              "prim::Constant"]
    assert len(consts) > 0
    some_const = consts[0]
    nc_decoder = TorchScriptPythonDecoder(model, some_const)
    ov_const = nc_decoder.as_constant()
    assert ov_const is not None
    assert len(ov_const) == 1
    assert ov_const[0].get_element_type() == Type.i16
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_i32_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    assert ov_const[0].get_element_type() == Type.i64
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_float_list():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    assert ov_const[0].get_element_type() == Type.i64
    assert ov_const[0].get_partial_shape() == PartialShape([2])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_float_tuple():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

    class aten_roll(torch.nn.Module):
        def __init__(self, shifts):
            super(aten_roll, self).__init__()
            self.shifts = shifts

        def forward(self, x):
            # roll has optional input dim, which is empty int list by default
            return torch.roll(x, self.shifts)

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
    assert ov_const[0].get_element_type() == Type.i64
    assert ov_const[0].get_partial_shape() == PartialShape([0])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_int_scalar_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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
    assert ov_const[0].get_element_type() == Type.i64
    assert ov_const[0].get_partial_shape() == PartialShape([])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_float_scalar_tensor():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type

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


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_tensor_list():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from openvino import PartialShape, Type
    from typing import List, Optional

    class SomeTensor(torch.nn.Module):
        def forward(self):
            l = torch.jit.annotate(List[Optional[torch.Tensor]], [
                                   torch.ones((1, 3, 3), dtype=torch.float),])
            return l

    model = get_scripted_model(SomeTensor())
    consts = list(model.graph.findAllNodes("prim::Constant"))
    assert len(consts) == 1, "Input model should contain 1 prim::Constant"
    nc_decoder = TorchScriptPythonDecoder(model)
    graph = nc_decoder.graph_element
    converted_const_nodes = list(graph.findAllNodes("prim::Constant"))
    converted_listconstruct_nodes = list(
        graph.findAllNodes("prim::ListConstruct"))
    # # Assert that replaced const exist and is not used
    assert len(converted_const_nodes) == 2
    assert len(
        [node for node in converted_const_nodes if not node.hasUses()]) == 1
    # Assert that prim::ListConstruct exist and has uses
    assert len(converted_listconstruct_nodes) == 1
    assert converted_listconstruct_nodes[0].kind() == "prim::ListConstruct"
    assert converted_listconstruct_nodes[0].hasUses()
    assert len(list(converted_listconstruct_nodes[0].inputs())) == 1
    created_const = converted_listconstruct_nodes[0].input().node()
    assert created_const in converted_const_nodes
    created_const_decoder = TorchScriptPythonDecoder(
        model, created_const).as_constant()
    assert created_const_decoder[0].get_element_type() == Type.f32
    assert created_const_decoder[0].get_partial_shape() == PartialShape([
        1, 3, 3])


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_tensor_list_empty():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from typing import List, Optional

    class SomeTensor(torch.nn.Module):
        def forward(self):
            l = torch.jit.annotate(List[Optional[torch.Tensor]], [])
            return l

    model = get_scripted_model(SomeTensor())
    consts = list(model.graph.findAllNodes("prim::Constant"))
    assert len(consts) == 1, "Input model should contain 1 prim::Constant"
    nc_decoder = TorchScriptPythonDecoder(model)
    graph = nc_decoder.graph_element
    converted_const_nodes = list(graph.findAllNodes("prim::Constant"))
    converted_listconstruct_nodes = list(
        graph.findAllNodes("prim::ListConstruct"))
    # Assert that replaced const exist and is not used
    assert len(converted_const_nodes) == 1
    assert not converted_const_nodes[0].hasUses()
    # Assert that prim::ListConstruct exist, has uses and dont have inputs
    assert len(converted_listconstruct_nodes) == 1
    assert converted_listconstruct_nodes[0].kind() == "prim::ListConstruct"
    assert converted_listconstruct_nodes[0].hasUses()
    assert len(list(converted_listconstruct_nodes[0].inputs())) == 0


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_optional_tensor_none():
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    from typing import Optional

    class SomeTensor(torch.nn.Module):
        def forward(self):
            l = torch.jit.annotate(Optional[torch.Tensor], None)
            return l

    model = get_scripted_model(SomeTensor())
    consts = list(model.graph.findAllNodes("prim::Constant"))
    assert len(consts) == 1, "Input model should contain 1 prim::Constant"
    nc_decoder = TorchScriptPythonDecoder(model)
    graph = nc_decoder.graph_element
    converted_const_nodes = list(graph.findAllNodes("prim::Constant"))
    removed_consts = [
        node for node in converted_const_nodes if not node.hasUses()]
    created_consts = [node for node in converted_const_nodes if node.hasUses()]
    assert len(removed_consts) == len(created_consts) == 1
    # Assert that unused const has torch.OptionalType dtype
    assert isinstance(removed_consts[0].output().type(), torch.OptionalType)
    # Assert that replacer const has correct dtype
    assert isinstance(created_consts[0].output().type(), torch.NoneType)
    # Assert that graph has correct output
    outputs = list(nc_decoder.graph_element.outputs())
    assert len(outputs) == 1
    assert isinstance(outputs[0].type(), torch.NoneType)


def f(x, y):
    return x + y


@pytest.mark.precommit
def test_pytorch_decoder_can_convert_scripted_function():
    from openvino import convert_model, Type
    scripted = torch.jit.script(f)
    model = convert_model(scripted, input=[Type.f32, Type.f32])
    assert model is not None


@pytest.mark.precommit
@pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.4.0"),
                    reason="Unsupported on torch<2.4.0")
def test_pytorch_fx_decoder_extracts_signature():
    from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder

    class TestModel(torch.nn.Module):
        def forward(self, a, b):
            return a["x"] + a["y"] + b

    example = ({"x": torch.tensor(1), "y": torch.tensor(2)}, torch.tensor(3))
    em = torch.export.export(TestModel(), example)
    nc_decoder = TorchFXPythonDecoder(em.module())
    assert nc_decoder.get_input_signature_name(0) == "a"
    assert nc_decoder.get_input_signature_name(1) == "b"
    assert nc_decoder._input_signature == ["a", "b"]
