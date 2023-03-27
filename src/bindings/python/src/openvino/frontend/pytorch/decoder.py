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
import inspect

def make_constant(*args, **kwargs):
    return op.Constant(*args, **kwargs)

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
        if ivalue.ndim == 0:
            assert str(ivalue.dtype()) in pt_to_ov_type_map, f"Type is not known {ivalue.dtype()}"
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
    float: OVType.f32,
    int: OVType.i32,
    "bool": OVType.boolean,
    "torch.float16": OVType.f16,
    "torch.float32": OVType.f32,
    torch.float32: OVType.f32,
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
            return self._as_constant_tensor(pt_value)
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
    def _as_constant_tensor(pt_value: torch.Value):
        ivalue = pt_value.toIValue()
        if pt_value.isCompleteTensor():
            if ivalue.ndim == 0:
                assert str(ivalue.dtype) in pt_to_ov_type_map, f"Type is not known {ivalue.dtype}"
                ov_type = pt_to_ov_type_map[str(ivalue.dtype)]
                ov_const = op.Constant(ov_type, Shape([]), [ivalue.item()])
            else:
                ivalue = ivalue.to(memory_format=torch.contiguous_format)
                narr = ivalue.numpy(force=True)
                if not narr.flags['C_CONTIGUOUS']:
                    narr = np.ascontiguousarray(narr)
                # Constant interpretation doesn't respect new-full type of PT
                # It recognizes only tensors, and give lists as 1D tensors, and scalars as Tensor scalars
                # So only tensor-type constants are supported
                ov_const = op.Constant(narr, shared_memory=True)
            return ov_const.outputs()
        else:
            return ivalue_to_constant(ivalue)
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

    def inlined_inputs(self, index):
        return []

class TorchFXPythonDecoder (Decoder):

    def __init__(self, pt_module, fx_gm, nodes=None, mark_node_callback=None, input_shapes=[], input_types=[]):
        Decoder.__init__(self)
        self.mark_node_callback = mark_node_callback
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self.pt_module = pt_module
        self.fx_gm = fx_gm
        self.input_types = input_types
        self.input_shapes = input_shapes

        print(type(pt_module))
        if issubclass(type(pt_module), torch.fx.graph_module.GraphModule):

            print(f'[ FX DECODER DEBUG ] __init__ called for GraphModule')

            self._nodes = list(pt_module.graph.nodes)
            self._inputs = []
            self._outputs = []
            for i in range(len(self._nodes)):
                if self._nodes[i].op == 'placeholder':
                    self._inputs.append(i)
                elif self._nodes[i].op == 'output':
                    # Instead of putting output index, refer to its target
                    #print(dir(self._nodes[i]))
                    args = self._nodes[i].args
                    if isinstance(args[0], tuple):
                        args = args[0]
                    print(args)
                    for output in args:
                        self._outputs.append(self._nodes.index(output))

            print(f'[ FX DECODER DEBUG ] inputs: {self._inputs}')
            print(f'[ FX DECODER DEBUG ] outputs: {self._outputs}')
            print(f'[ FX DECODER DEBUG ] #nodes: {len(self._nodes)}')

        elif issubclass(type(pt_module), torch.fx.Node):

            print(f'[ FX DECODER DEBUG ] __init__ called for Node')

            self._nodes = nodes # passed from outer context
            print(f'len(nodes) = {len(nodes)}')

            # FIXME: Quadratic complexity nodes*nodes considering the outer loop over all nodes
            for i in range(len(self._nodes)):
                if self._nodes[i] == pt_module:
                    self._outputs = [i]

            print(f'pt_module.args = {pt_module.args}')
            # code constant or None input as a tuple to diffirentiate it from regular input
            # negative index is used to mark inputs initialized by inline constants, there are no such inputs in the graph
            self._inputs = [self._nodes.index(arg) if arg in self._nodes else (arg,) for arg in pt_module.args]

            print(f'[ FX DECODER DEBUG ] inputs: {self._inputs}')
            print(f'[ FX DECODER DEBUG ] outputs: {self._outputs}')


    def inputs(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        print([x if x is not None else 100000 for x in self._inputs])
        return [x if x is not None else 100000 for x in self._inputs]

    def input(self, index):  # TODO: remove
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return self.inputs()[index]  # TODO: find specialized method

    def get_input_debug_name(self, index):
        return "input"+str(index)

    def get_input_shape(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if index < len(self.input_shapes):
            return PartialShape(self.input_shapes[index])
        input = self._raw_input(index)
        return self.get_shape_for_value(input)

    def get_input_type(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if index < len(self.input_types):
            return OVAny(pt_to_ov_type_map[self.input_types[index]])
        input = self._raw_input(index)
        return self.get_type_for_value(input)

    def get_output_debug_name(self, index):
        return "output"+str(index)

    def get_output_shape(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def _get_known_type_for_value(self, type):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        '''
            Returns known/unknown types wrapped as OVAny
        '''
        #print(f'Trying to parse type {type} of class {type.__class__}')
        # Check for simple scalar types first
        # TODO: Don't use str, use native types
        if type is None:
            return OVAny(OVType.dynamic)
        if str(type) in pt_to_ov_type_map:
            #print(f'Recognized native type, type.__class__ = {type.__class__}')
            return OVAny(pt_to_ov_type_map[str(type)])
        elif type.__class__ is torch.TensorType:
            #print(f'Recognized Tensor type with type.dtype() = {type.dtype()}')
            # Tensor type, parse element type
            # TODO: replace string by native type
            # return OVAny(PartialShape([1,2,3]))
            return OVAny(DecoderType.Tensor(self._get_known_type_for_value(type.dtype())))
        elif type.__class__ is torch.ListType:
            element_type = type.getElementType()
            #print(f'Recognized torch List type. Type of element is {element_type}')
            return OVAny(DecoderType.List(self._get_known_type_for_value(element_type)))
        else:
            #print(f'Not a tensor nor native type: {type}')
            # Not yet recognized
            return OVAny(OVType.dynamic)
            #pt_type_class = value.type().__class__
            #    if pt_type_class is torch.ListType:

    def get_shape_for_value(self, value):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if value and ('tensor_meta' in value.meta.keys()):
            return PartialShape(value.meta['tensor_meta'].shape)
        return PartialShape([1])

    def get_type_for_value(self, value):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if value and ('tensor_meta' in value.meta.keys()):
            pt_type = value.meta['tensor_meta'].dtype
            if pt_type in pt_to_ov_type_map:
                ov_type = pt_to_ov_type_map[pt_type]
                return OVAny(ov_type)
        return OVAny(OVType.f32)

    def get_input_transpose_order(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return []
        # TODO TBD

        input = self._raw_input(index)
        if input.type() is not None and input.type().kind() == 'TensorType':
            strides = input.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_output_transpose_order(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return []

        # old code
        output = self._raw_output(index)
        if output.type() is not None and output.type().kind() == 'TensorType':
            strides = output.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_subgraph_size(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if issubclass(type(self.pt_module), torch.fx.Node):
            return 0
        return len(self.get_subgraphs()) if hasattr(self.pt_module, 'blocks') else 1

    def visit_subgraph(self, node_visitor):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        # make sure topological order is satisfied
        for node in self._nodes:
            if node.op == 'placeholder' or node.op == 'output':
                continue # skipping non-operational nodes
            decoder = TorchFXPythonDecoder(node, self.fx_gm, self._nodes, mark_node_callback=self.mark_node_callback)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def get_subgraphs(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if issubclass(type(self.pt_module), torch.fx.Node):
            return []
        return list(self.pt_module.blocks())

    def get_subgraph_decoder(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        decoder = TorchFXPythonDecoder(self.get_subgraphs()[index], self.fx_gm, mark_node_callback=self.mark_node_callback)
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        print(f'fx.op = {self.pt_module.op}')
        if self.pt_module.op == 'call_function':
            print(f'function: {self.pt_module.target} str of it: {str(self.pt_module.target)}')
            return str(self.pt_module.target)
        elif self.pt_module.op == 'get_attr':
            return 'get_attr'  # FIXME should be aligned with get_attr from TS implementation
        else:
            print('UNKNOWN_TYPE')
            return 'UNKNOWN_TYPE_' + str(self.pt_module.op)

    def get_schema(self):
        return ''
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return self.pt_module.schema()

    def outputs(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return self._outputs

    def _raw_outputs(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return [self._nodes[x] for x in self._outputs]

    def _raw_output(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return self._raw_outputs()[index]

    def _raw_inputs(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        #print([self._nodes[x] if x is not None else None for x in self._inputs])
        return [self._nodes[x] if x is not None and x < len(self._nodes) else x for x in self._inputs]

    def _raw_input(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return self._raw_inputs()[index]

    def num_of_outputs(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return len(self.outputs())

    def output(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        return self.outputs()[index]

    def mark_node(self, node):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if self.mark_node_callback is not None:
            self.mark_node_callback(self, node)
        return node

    def as_constant(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')

        if self.pt_module.op == 'get_attr':
            # Extract Constant from FX module field
            print(dir(self.pt_module))
            ret = fetch_attr(self.fx_gm, self.pt_module.target)
            print(type(ret))
            print(ret.size())
            ovshape = PartialShape(ret.size())
            ovtype = pt_to_ov_type_map[ret.type()]
            print(ovshape, ovtype)
            ov_const = make_constant(ovtype, ovshape.get_shape(), ret.data_ptr())
            print('Made constant')
            return ov_const.outputs()


        if not self.get_op_type() == 'prim::Constant':
            #print(f'[ ERROR ] Requested const value {self._raw_output(0)} from a non const prim {self.get_op_type()}')
            return None
        pt_value = self._raw_output(0)

        pt_type_class = pt_value.type().__class__
        #print(f'Not a tensor, type = {pt_value.type()}\ndir = {dir(pt_value.type())}\n__class__ = {pt_value.type().__class__}')
        if pt_type_class is torch.TensorType:
            return self.as_constant_tensor(pt_value)
        if pt_type_class is torch.ListType:
            return self.as_constant_list(pt_value)
        #print(f'Trying to recognize value {pt_value}, type = {type(pt_value.toIValue())}, ivalue = {pt_value.toIValue()}')
        if str(pt_value.type()) in ['torch.int32', 'int']:
            #print(f'Found int value=  {pt_value}, type = {type(pt_value.toIValue())}, ivalue = {pt_value.toIValue()}')
            return make_constant(OVType.i32, Shape([]), [pt_value.toIValue()]).outputs()
        if str(pt_value.type()) in ['torch.float', 'torch.FloatType', 'float']:
            #print(f'Found float value=  {pt_value}, type = {type(pt_value.toIValue())}, ivalue = {pt_value.toIValue()}')
            return make_constant(OVType.f32, Shape([]), [pt_value.toIValue()]).outputs()
        if str(pt_value.type()) in ['torch.bool', 'bool']:
            #print('Scalar bool detected')
            return make_constant(OVType.boolean, Shape([]), [pt_value.toIValue()]).outputs()
        #print(f'Left value not converted to const, value = {pt_value}')

        return None

    def as_string(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        if not self.get_op_type() == 'prim::Constant':
            return None
        pt_value = self._raw_output(0)

        if str(pt_value.type()) in ['torch.StringType', 'str']:
            return pt_value.toIValue()
        return None

    def as_constant_tensor(self, pt_value):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_core.co_name}')
        ivalue = pt_value.toIValue()
        print(f'[ FX DECODER DEBUG ] as_constant_tensor with {pt_value}')
        if pt_value.isCompleteTensor():
            try:
                ivalue = ivalue.to(memory_format=torch.contiguous_format).detach().cpu()
            except:
                print("[ WARNING ] Tensor couldn't detach")
            if str(pt_value.type().dtype()) in pt_to_py_type_map:
                # Constant interpretation doesn't respect new-full type of PT
                # It recognizes only tensors, and give lists as 1D tensors, and scalars as Tensor scalars
                # So only tensor-type constants are supported
                ovshape = PartialShape(pt_value.type().sizes())
                ovtype = pt_to_ov_type_map[str(pt_value.type().dtype())]

                # TODO: try-except here is a temporary WA for issues with data_ptr that we currently cannot predict; provide better solution
                try:
                    # this is only possible with adding a new ctor for Constant Python binding
                    # TODO Check strides and pass them somehow
                    values = ivalue.data_ptr()
                    ov_const = make_constant(ovtype, ovshape.get_shape(), values)
                except:
                    # old variant that makes a slow data copying
                    print(f"[ WARNING ] Constant wasn't able to convert from data_ptr.")
                    values = ivalue.flatten().tolist()
                    ov_const = make_constant(ovtype, ovshape.get_shape(), values)
                return ov_const.outputs()
        else:
            # Incomplete tensor can be scalar
            if isinstance(ivalue, float):
                return make_constant(OVType.f32, Shape([]), [ivalue]).outputs()
            if isinstance(ivalue, int):
                return make_constant(OVType.i32, Shape([]), [ivalue]).outputs()
            if isinstance(ivalue, bool):
                return make_constant(OVType.boolean, Shape([]), [ivalue]).outputs()

            # TODO: verify that it correctly reads incomplete consts
            if ivalue.type() in pt_to_ov_type_map:
                try:
                    ovshape = PartialShape(ivalue.size())
                    ovtype = pt_to_ov_type_map[ivalue.type()]
                    ov_const = make_constant(ovtype, ovshape.get_shape(), ivalue.data_ptr())
                except:
                    # old variant that makes a slow data copying
                    print(f"[ WARNING ] Constant wasn't able to convert from data_ptr.")
                    nvalues = ivalue.numpy()
                    ovtype = np_to_ov_type_map[str(nvalues.dtype)]
                    ovshape = PartialShape(nvalues.shape)
                    ov_const = make_constant(ovtype, ovshape.get_shape(), nvalues.flatten().tolist())
                return ov_const.outputs()
        return None

    def as_constant_list(self, pt_value):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        # For now it is treat a list as a 1D tensor; it is required by converters to avoid need to massively rewrite them in that part where constant attributes are queried
        pt_element_type = str(pt_value.type().getElementType())
        ivalue = pt_value.toIValue()
        #print(f'List toIValue: {ivalue}, type of it: {type(ivalue)}')
        is_known_type = pt_element_type in pt_to_ov_type_map

        # WA to broken ov.Type
        # Detect integer list and process it with a dedicated method
        # TODO: Fix ov.Type and remove this WA
        # if pt_to_py_type_map[pt_element_type] == 'int':
        #    self.as_constant_list_of_ints(ovshape = PartialShape([len(ivalue)]), ivalue)
        # End of WA to broken ov.Type

        if is_known_type:
            ovtype = pt_to_ov_type_map[pt_element_type]
            #print(f'ovtype = {ovtype}, pt_element_type = {pt_element_type}, OVType.i32 = {OVType.i32}, {OVType.f32}')
            ovshape = PartialShape([len(ivalue)])
            ov_const = make_constant(ovtype, ovshape.get_shape(), ivalue)
            return ov_const.outputs()

    def input_is_none(self, index):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        print(f'index = {index}, op = {self.get_op_type()}')
        if index >= len(self.inputs()) or self._raw_input(index) is None:
            print(f'saying None, {self._raw_input(index) if index < len(self.inputs()) else "out of range"}')
            return True
        else:
            r_input = self._raw_input(index)
            print(f'r_input = {r_input}')
            return str(type(r_input)) in ['torch.NoneType', 'NoneType']

    def debug(self):
        print(f'[ FX DECODER DEBUG ] Decoder method called: {inspect.currentframe().f_code.co_name}')
        print(f'DEBUG CALLED FOR {self._raw_output(0)}')
        # self.pt_module.print()

    def inlined_inputs(self, index):
        result = []
        for i in range(len(self._inputs)):
            if isinstance(self._inputs[i], tuple):
                constant = None
                arg = self._inputs[i][0]
                if isinstance(arg, list):
                    if len(arg) > 0:
                        constant = make_constant(pt_to_ov_type_map[type(arg[0])], Shape([len(arg)]), arg)
                    else:
                        # TODO: which type should we use if list is empty? Need a signaling value here
                        constant = make_constant(int, Shape([0]), [])
                elif isinstance(arg, bool):
                    constant = make_constant(OVType.boolean, Shape([]), [arg])
                elif isinstance(arg, int):
                    constant = make_constant(OVType.i32, Shape([]), [arg])  # TODO: i32? why not i64?
                elif isinstance(arg, float):
                    constant = make_constant(OVType.f32, Shape([]), [arg])  # TODO: f32? why not f64?

                if constant is None:
                    print(f'[ ERROR ] Not supported inlined input type {type(arg)} with value {arg}')
                    if arg is None:
                        self._inputs[i] = None
                else:
                    assert len(constant.outputs()) == 1
                    result.append(constant.outputs()[0])
                    self._inputs[i] = index
                    index += 1
        return result
