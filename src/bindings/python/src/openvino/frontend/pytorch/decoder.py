# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino.runtime import PartialShape, Type as OVType, OVAny, Shape

from openvino.runtime import op
import numpy as np
import torch


def make_constant(*args, **kwargs):
    return op.Constant(*args, **kwargs)


pt_to_ov_type_map = {
    'float': OVType.f32,
    'int': OVType.i32,
    'torch.float32': OVType.f32,
    'torch.int32': OVType.i32,
    "torch.bool": OVType.boolean,
    "torch.int64": OVType.i64,
    "torch.FloatTensor": OVType.f32,
    "torch.IntTensor": OVType.i32,
    "torch.LongTensor": OVType.i64,
    "torch.BoolTensor": OVType.boolean
}

pt_to_py_type_map = {
    'float': 'float',
    'int': 'int',
    'torch.float32': 'float',
    'torch.int32': 'int',
    'torch.int64': 'int',
    'torch.bool': 'bool'
}

np_to_ov_type_map = {
    'float32': OVType.f32,
    'int32': OVType.i32,
}


class TorchScriptPythonDecoder (Decoder):
    def __init__(self, pt_module):
        Decoder.__init__(self)
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self.pt_module = pt_module

    def inputs(self):
        return [x.unique() for x in self.pt_module.inputs()]

    def input(self, index):  # TODO: remove
        return self.inputs()[index]  # TODO: find specialized method

    def get_input_shape(self, index):
        input = self._raw_input(index)
        return self.get_shape_for_value(input)

    def get_input_type(self, index):
        input = self._raw_input(index)
        return self.get_type_for_value(input)

    def get_output_shape(self, index):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def _get_known_type_for_value(self, type):
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
        if value.isCompleteTensor():
            ps = PartialShape(value.type().sizes())
            #print(f'SHAPE FOR COMPLETE TENSOR: {ps}')
            return ps
        else:
            #print(f'NOT COMPLETE TENSOR for {value}')
            # TODO: Recognize types that we can represent as a nested constructs with objects from DecoderType
            # If recognized, return scalar instead of dynamic. Scalar means a single value of that custom type.
            # See get_type_for_value for reference
            pass
        return PartialShape.dynamic()

    def get_type_for_value(self, value):
        #print(f'Decoding value type for value {value}')
        full_type = self._get_known_type_for_value(value.type())
        # DecoderType.print(full_type)    # new (full) type interpretation
        return full_type
        # Old version of this function directly treat Tensor[type] as type
        # assuming that regular type for vaue is Tensor, so it just
        # decodes its element type.
        # In full_type we code a complete type according to PT, it allows
        # to distiguish int from scalar Tensor[int] in particular.
        # It is necessary to interpreting some operations converting scalar values (not tensors)
        # to scalar tensors.
        # In this new interpretation we leave old beheviout to FE code if it is still needed
        if value.isCompleteTensor():
            pt_type = str(value.type().dtype())
            print(f'Trying to decode tensor element type: {pt_type}')
            if pt_type in pt_to_ov_type_map:
                ov_type = pt_to_ov_type_map[pt_type]
                print(f'[ DEBUG ] Decoded ov type: {ov_type}', flush=True)
                return OVAny(ov_type)
            else:
                print(f'[ DEBUG ] Unrecognized pt element type for a tensor: {pt_type}. Captured it as custom type.', flush=True)
                # TODO: Replace it by Tensor[dynamic]
                return OVAny(OVType.dynamic)
        else:
            return OVAny(OVType.dynamic)

    def get_input_transpose_order(self, index):
        input = self._raw_input(index)
        if input.type() is not None and input.type().kind() == 'TensorType':
            strides = input.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_output_transpose_order(self, index):
        output = self._raw_output(index)
        if output.type() is not None and output.type().kind() == 'TensorType':
            strides = output.type().strides()
            if strides is not None:
                return [s[0] for s in sorted(enumerate(strides), key=lambda x:x[1], reverse=True)]
        return []

    def get_subgraph_size(self):
        return len(self.get_subgraphs()) if hasattr(self.pt_module, 'blocks') else 1

    def visit_subgraph(self, node_visitor):
        # make sure topological order is satisfied
        for node in self.pt_module.nodes():
            decoder = TorchScriptPythonDecoder(node)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def get_subgraphs(self):
        return list(self.pt_module.blocks())

    def get_subgraph_decoder(self, index):
        decoder = TorchScriptPythonDecoder(self.get_subgraphs()[index])
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self):
        return self.pt_module.kind()

    def get_schema(self):
        return self.pt_module.schema()

    def outputs(self):
        return [x.unique() for x in self.pt_module.outputs()]

    def _raw_outputs(self):
        return [x for x in self.pt_module.outputs()]

    def _raw_output(self, index):
        return self._raw_outputs()[index]

    def _raw_inputs(self):
        return [x for x in self.pt_module.inputs()]

    def _raw_input(self, index):
        return self._raw_inputs()[index]

    def num_of_outputs(self):
        return len(self.outputs())

    def output(self, index):
        return self.outputs()[index]

    def mark_node(self, node):
        return node

    def as_constant(self):
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
        if not self.get_op_type() == 'prim::Constant':
            return None
        pt_value = self._raw_output(0)

        if str(pt_value.type()) in ['torch.StringType', 'str']:
            return pt_value.toIValue()
        return None

    def as_constant_tensor(self, pt_value):
        ivalue = pt_value.toIValue()
        if pt_value.isCompleteTensor():            
            try:
                ivalue = ivalue.to(memory_format=torch.contiguous_format).detach().cpu()
            except:
                print("[ WARNING ] Tensor couldn't detach")
            if str(pt_value.type().dtype()) in pt_to_ov_type_map:
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
        if index >= len(self.inputs()) or self._raw_input(index) is None:
            return True
        else:
            r_input = self._raw_input(index)
            return str(r_input.type()) in ['torch.NoneType', 'NoneType']

    def debug(self):
        print(f'DEBUG CALLED FOR {self._raw_output(0)}')
        # self.pt_module.print()
