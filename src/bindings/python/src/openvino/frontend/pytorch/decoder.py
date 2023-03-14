# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape

import typing
import warnings
import torch
import numpy as np


def get_type_from_py_type(value):
    if isinstance(value, float):
        return OVType.f32
    if isinstance(value, bool):
        return OVType.boolean
    if isinstance(value, int):
        # Python int is 64 bit, but we will convert it to int32 except cases when it can't fit in 32 bits
        if torch.iinfo(torch.int).min <= value <= torch.iinfo(torch.int).max:
            return OVType.i32
        return OVType.i64
    return OVType.dynamic


def ivalue_to_constant(ivalue):
    ov_type = get_type_from_py_type(ivalue)
    if ov_type.is_static():
        return op.Constant(ov_type, Shape([]), [ivalue]).outputs()

    if isinstance(ivalue, (list, tuple)):
        assert len(ivalue) > 0, "Can't deduce type for empty list"
        ov_type = get_type_from_py_type(ivalue[0])
        assert ov_type.is_static(), "Can't deduce type for list"
        return op.Constant(ov_type, Shape([len(ivalue)]), ivalue).outputs()

    if isinstance(ivalue, torch.Tensor):
        if ivalue.dim() == 0:
            assert str(ivalue.dtype) in pt_to_ov_type_map, f"Type is not known {ivalue.dtype}"
            ov_type = pt_to_ov_type_map[str(ivalue.dtype)]
            ov_const = op.Constant(ov_type, Shape([]), [ivalue.item()])
        else:
            ivalue = ivalue.to(memory_format=torch.contiguous_format)
            narr = ivalue.numpy(force=True)
            if not narr.flags['C_CONTIGUOUS']:
                narr = np.ascontiguousarray(narr)
            ov_const = op.Constant(narr, shared_memory=True)
        return ov_const.outputs()
    return None


def get_value_from_getattr(getattr_node, self_module):
    assert getattr_node.kind() == "prim::GetAttr", "Got node of kind not equal to prim::GetAttr"
    # GetAttr nodes can be nested
    stack = []
    while getattr_node.kind() == "prim::GetAttr":
        stack.append(getattr_node)
        inputs = list(getattr_node.inputs())
        if len(inputs) == 0:
            break
        getattr_node = inputs[0].node()
    module = self_module
    while len(stack) > 0:
        node = stack.pop()
        assert (hasattr(module, node.s("name")))
        module = getattr(module, node.s("name"))
    return module


pt_to_ov_type_map = {
    "float": OVType.f32,
    "int": OVType.i32,
    "bool": OVType.boolean,
    "torch.float16": OVType.f16,
    "torch.float32": OVType.f32,
    "torch.float64": OVType.f64,
    "torch.uint8": OVType.u8,
    "torch.int8": OVType.i8,
    "torch.int32": OVType.i32,
    "torch.int64": OVType.i64,
    "torch.bool": OVType.boolean,
    "torch.DoubleTensor": OVType.f64,
    "torch.FloatTensor": OVType.f32,
    "torch.IntTensor": OVType.i32,
    "torch.LongTensor": OVType.i64,
    "torch.BoolTensor": OVType.boolean,
}


class TorchScriptPythonDecoder (Decoder):
    def __init__(self, pt_module, graph_element=None):
        Decoder.__init__(self)
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        if graph_element is None:
            assert hasattr(pt_module, "inlined_graph"), "graph_element must have inlined_graph"
            self.graph_element = pt_module.inlined_graph
        else:
            self.graph_element = graph_element
        self.pt_module = pt_module
        self.raw_inputs = list(self.graph_element.inputs())
        self.raw_outputs = list(self.graph_element.outputs())

    def inputs(self) -> list:
        return [x.unique() for x in self.raw_inputs]

    def get_input(self, index: int):
        return self.inputs()[index]

    def get_input_debug_name(self, index: int) -> str:
        return self._raw_input(index).debugName()

    def get_input_shape(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_shape_for_value(raw_input)

    def get_input_type(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_type_for_value(raw_input)

    def get_output_debug_name(self, index: int) -> str:
        return self._raw_output(index).debugName()

    def get_output_shape(self, index: int):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index: int):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def _get_known_type_for_value(self, pt_type):
        """Returns known/unknown types wrapped as OVAny."""
        # Check for simple scalar types first
        if pt_type is None:
            return OVAny(OVType.dynamic)
        # TODO: Don't use str, use native types
        if str(pt_type) in pt_to_ov_type_map:
            return OVAny(pt_to_ov_type_map[str(pt_type)])
        elif isinstance(pt_type, torch.TensorType):
            # Tensor type, parse element type
            return OVAny(DecoderType.Tensor(self._get_known_type_for_value(pt_type.dtype())))
        elif isinstance(pt_type, torch.ListType):
            element_type = pt_type.getElementType()
            return OVAny(DecoderType.List(self._get_known_type_for_value(element_type)))
        elif isinstance(pt_type, (torch.StringType, torch.DeviceObjType)):
            return OVAny(DecoderType.Str())
        elif isinstance(pt_type, torch.NoneType):
            return OVAny(DecoderType.PyNone())
        else:
            # Not yet recognized
            return OVAny(OVType.dynamic)

    def get_shape_for_value(self, value: torch.Value):
        if value.isCompleteTensor():
            ps = PartialShape(value.type().sizes())
            return ps
        else:
            # TODO: Recognize types that we can represent as a nested constructs with objects from DecoderType
            # If recognized, return scalar instead of dynamic. Scalar means a single value of that custom type.
            # See get_type_for_value for reference
            pass
        return PartialShape.dynamic()

    def get_type_for_value(self, value: torch.Value):
        full_type = self._get_known_type_for_value(value.type())
        return full_type

    def get_input_transpose_order(self, index: int) -> list:
        raw_input = self._raw_input(index)
        if raw_input.type() is not None and raw_input.type().kind() == "TensorType":
            strides = raw_input.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_output_transpose_order(self, index: int) -> list:
        output = self._raw_output(index)
        if output.type() is not None and output.type().kind() == "TensorType":
            strides = output.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_subgraph_size(self) -> int:
        if isinstance(self.graph_element, torch.Node):
            return len(self.get_subgraphs()) 
        else:
            return 1

    def visit_subgraph(self, node_visitor) -> None:
        # make sure topological order is satisfied
        for node in self.graph_element.nodes():
            decoder = TorchScriptPythonDecoder(self.pt_module, node)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def get_subgraphs(self) -> list:
        if self.graph_element.kind() == "prim::PythonOp":
            if "Subgraph" in self.graph_element.attributeNames():
                assert isinstance(self.graph_element, torch.Node), "Graph element must be of type torch.Node."
                return [getattr(self.graph_element, self.graph_element.kindOf("Subgraph"))("Subgraph")]
            else:
                # Attribute "Subgraph" is only available if Graph was created using tracing.
                # TODO Find way to extract subgraph for scripted Graph.
                return []
        return list(self.graph_element.blocks())

    def get_subgraph_decoder(self, index: int):
        decoder = TorchScriptPythonDecoder(self.pt_module, self.get_subgraphs()[index])
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self) -> str:
        return self.graph_element.kind()

    def get_schema(self) -> str:
        return self.graph_element.schema()

    def outputs(self) -> list:
        return [x.unique() for x in self.raw_outputs]

    def _raw_output(self, index: int):
        return self.raw_outputs[index]

    def _raw_input(self, index: int):
        return self.raw_inputs[index]

    def num_of_outputs(self):
        return len(self.raw_outputs)

    def output(self, index: int):
        return self.outputs()[index]

    def mark_node(self, node):
        return node

    def try_decode_get_attr(self):
        pt_value = get_value_from_getattr(self.graph_element, self.pt_module)
        assert pt_value is not None, "Couldn't retrieve value from prim::GetAttr"
        if not isinstance(pt_value, (torch.jit.ScriptModule, torch.jit.TracedModule)):
            return ivalue_to_constant(pt_value)
        else:
            return []

    def as_constant(self):
        if not self.get_op_type() == "prim::Constant":
            return None
        pt_value = self._raw_output(0)

        pt_type = pt_value.type()
        if isinstance(pt_type, torch.TensorType):
            return ivalue_to_constant(pt_value.toIValue())
        if isinstance(pt_type, torch.ListType):
            return self._as_constant_list(pt_value)
        return ivalue_to_constant(pt_value.toIValue())

    def as_string(self):
        if self.get_op_type() == "prim::Constant":
            pt_value = self._raw_output(0)
            if str(pt_value.type()) in ["torch.StringType", "str"]:
                return pt_value.toIValue()
            elif str(pt_value.type()) == "Device":
                return pt_value.toIValue().type
        elif self.get_op_type() == "prim::device":
            return self._get_device_string()
        return None

    @staticmethod
    def _as_constant_list(pt_value: torch.Value):
        # For now it is treat a list as a 1D tensor; it is required by converters to avoid need to massively
        # rewrite them in that part where constant attributes are queried
        pt_element_type = str(pt_value.type().getElementType())
        ivalue = pt_value.toIValue()
        is_known_type = pt_element_type in pt_to_ov_type_map

        if is_known_type:
            ovtype = pt_to_ov_type_map[pt_element_type]
            ovshape = PartialShape([len(ivalue)])
            ov_const = op.Constant(ovtype, ovshape.get_shape(), ivalue)
            return ov_const.outputs()

    def _get_device_string(self) -> str:
        assert self.graph_element.kind() == "prim::device", "This function can be called for prim::device node."
        value = self.raw_inputs[0]
        if value.type().isSubtypeOf(torch.TensorType.get()):
            tensor = typing.cast(torch.TensorType, value.type())
            device = tensor.device()
            if device:
                return str(device)
        # Device cannot be statically determined.
        return "cpu"

    def input_is_none(self, index: int) -> bool:
        if index >= len(self.inputs()) or self._raw_input(index) is None:
            return True
        else:
            r_input = self._raw_input(index)
            if str(r_input.type()) in ["torch.NoneType", "NoneType"]:
                return True
            else:
                in_node = r_input.node()
                if in_node.kind() == "prim::GetAttr":
                    pt_value = get_value_from_getattr(in_node, self.pt_module)
                    return pt_value is None
        return False
