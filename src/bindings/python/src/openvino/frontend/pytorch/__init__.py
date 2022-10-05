# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Package: openvino
Low level wrappers for the FrontEnd C++ API.
"""

# flake8: noqa

from openvino.utils import add_openvino_libs_to_path
from openvino.runtime import PartialShape, Type as OVType, OVAny, Shape

from openvino.runtime import op

add_openvino_libs_to_path()


try:
    from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
    from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
    import numpy as np
    import torch

    # TODO: remove this crazy stuff, it saves all decoders ever created
    # This is a WA for well known bug in pybind11 with too early deinitialization
    decoders = []

    pt_to_ov_type_map = {
        'float': OVType.f32,
        'int': OVType.i32,
        'torch.float32': OVType.f32,
        'torch.int32': OVType.i32
    }

    pt_to_py_type_map = {
        'float': 'float',
        'int': 'int',
        'torch.float32': 'float',
        'torch.int32': 'int'
    }

    class TorchScriptPythonDecoder (Decoder):
        def __init__ (self, pt_module):
            Decoder.__init__(self)
            self.pt_module = pt_module
            # TODO: remove this; leads to huge memory leaks while converting many models in one app
            decoders.append(self)
            #print(dir(pt_module))
            # Print schema for nodes if any
            #print(f'there is schema: {pt_module.schema()}' if hasattr(pt_module, 'schema') else 'no schema')
            #print(pt_module)
            #exit()

        def free_decoders(self):
            decoders.clear()
        
        def inputs (self):
            return [x.unique() for x in self.pt_module.inputs()]

        def input (self, index): # TODO: remove
            return self.inputs()[index] # TODO: find specialized method 

        def get_input_shape (self, index):
            input = self._raw_input(index)
            return self.get_shape_for_value(input)

        def get_input_type (self, index):
            input = self._raw_input(index)
            return self.get_type_for_value(input)


        def get_output_shape (self, index):
            output = self._raw_output(index)
            return self.get_shape_for_value(output)

        def get_output_type (self, index):
            output = self._raw_output(index)
            return self.get_type_for_value(output)

        def _get_known_type_for_value (self, type):
            #return
            '''
                Returns known/unknown types wrapped as OVAny
            '''
            print(f'Trying to parse type {type} of class {type.__class__}')
            # Check for simple scalar types first
            # TODO: Don't use str, use native types
            if str(type) in pt_to_ov_type_map:
                print(f'Recognized native type, type.__class__ = {type.__class__}')
                return OVAny(pt_to_ov_type_map[str(type)])
            elif type.__class__ is torch.TensorType:
                print(f'Recognized Tensor type with type.dtype() = {type.dtype()}')
                # Tensor type, parse element type
                # TODO: replace string by native type
                #return OVAny(PartialShape([1,2,3]))
                return OVAny(DecoderType.Tensor(self._get_known_type_for_value(type.dtype())))
            elif type.__class__ is torch.ListType:
                element_type = type.getElementType()
                print(f'Recognized torch List type. Type of element is {element_type}')
                return OVAny(DecoderType.List(self._get_known_type_for_value(element_type)))
            else:
                print(f'Not a tensor nor native type: {type}')
                # Not yet recognized
                return OVAny(OVType.dynamic)
                #pt_type_class = value.type().__class__
                #    if pt_type_class is torch.ListType:


        def get_shape_for_value (self, value):
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

        def get_type_for_value (self, value):
            print(f'Decoding value type for value {value}')
            full_type = self._get_known_type_for_value(value.type())
            DecoderType.print(full_type)    # new (full) type interpretation
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

        def get_input_transpose_order (self, index):
            return []

        def get_output_transpose_order (self, index):
            return []
        
        def get_subgraph_size (self):
            return len(self.get_subgraphs()) if hasattr(self.pt_module, 'blocks') else 1

        def visit_subgraph (self, index, node_visitor):
            # make sure topological order is satisfied
            if index < self.get_subgraph_size():
                if hasattr(self.pt_module, 'blocks'):
                    for node in self.get_subgraphs()[index].nodes():
                        #print('inside 1')
                        #print('visit node/block')
                        #print(node, flush=True)
                        decoder = TorchScriptPythonDecoder(node)
                        node_visitor(decoder)
                else:
                    for node in self.pt_module.nodes():
                        #print('inside 2')
                        #print('visit node/block')
                        #print(node, flush=True)
                        decoder = TorchScriptPythonDecoder(node)
                        node_visitor(decoder)
            else:
                raise Exception(f'Index {index} of block is out of range, total number of blocks is {self.get_subgraph_size()}')

        def get_subgraphs (self):
            return list(self.pt_module.blocks())

        def get_subgraph_decoder (self, index):
            return TorchScriptPythonDecoder(self.get_subgraphs()[index])

        def get_op_type (self):
            return self.pt_module.kind()

        def get_schema (self):
            return self.pt_module.schema()

        def outputs (self):
            return [x.unique() for x in self.pt_module.outputs()]

        def _raw_outputs (self):
            return [x for x in self.pt_module.outputs()]

        def _raw_output (self, index):
            return self._raw_outputs()[index]

        def _raw_inputs (self):
            return [x for x in self.pt_module.inputs()]

        def _raw_input (self, index):
            return self._raw_inputs()[index]

        def num_of_outputs (self):
            return len(self.outputs())

        def output (self, index):
            return self.outputs()[index]

        def mark_node (self, node):
            return node

        def as_constant (self):
            if not self.get_op_type() == 'prim::Constant':
                #print(f'[ ERROR ] Requested const value {self._raw_output(0)} from a non const prim {self.get_op_type()}')
                return None
            pt_value = self._raw_output(0)
            is_tensor = pt_value.isCompleteTensor()
            
            print(f'Decoding value type for constant value {pt_value}')
            DecoderType.print(self._get_known_type_for_value(pt_value.type()))

            if is_tensor and str(pt_value.type().dtype()) in pt_to_py_type_map:
                return self.as_constant_tensor(pt_value)
            
            if not is_tensor:
                pt_type_class = pt_value.type().__class__
                #print(f'Not a tensor, type = {pt_value.type()}\ndir = {dir(pt_value.type())}\n__class__ = {pt_value.type().__class__}')
                if pt_type_class is torch.ListType:
                    return self.as_constant_list(pt_value)
                #print(f'Trying to recognize value {pt_value}, type = {type(pt_value.toIValue())}, ivalue = {pt_value.toIValue()}')
                if str(pt_value.type()) in ['torch.int32', 'int']:
                    #print(f'Found int value=  {pt_value}, type = {type(pt_value.toIValue())}, ivalue = {pt_value.toIValue()}')
                    return op.Constant(OVType.i32, Shape([]), [pt_value.toIValue()]).outputs()
                if str(pt_value.type()) in ['torch.float', 'torch.FloatType', 'float']:
                    #print(f'Found float value=  {pt_value}, type = {type(pt_value.toIValue())}, ivalue = {pt_value.toIValue()}')
                    return op.Constant(OVType.f32, Shape([]), [pt_value.toIValue()]).outputs()
                if str(pt_value.type()) in ['torch.bool', 'bool']:
                    #print('Scalar bool detected')
                    return op.Constant(OVType.boolean, Shape([]), [pt_value.toIValue()]).outputs()
                print(f'Left value not converted to const, value = {pt_value}')
            else:
                print(f'Not a known type, dtype = {pt_value.type().dtype()}')

            return None

        def as_constant_tensor (self, pt_value):
            # Constant interpretation doesn't respect new-full type of PT
            # It recognizes only tensors, and give lists as 1D tensors, and scalars as Tensor scalars
            # So only tensor-type constants are supported
            ovshape = PartialShape(pt_value.type().sizes())
            ovtype = pt_to_ov_type_map[str(pt_value.type().dtype())]
            np_value = pt_value.toIValue().cpu().detach().numpy().flatten().tolist()  # TODO: find a better/shorter way
            ov_const = op.Constant(ovtype, ovshape.get_shape(), np_value)
            return ov_const.outputs()

        def as_constant_list (self, pt_value):
            # For now it is treat a list as a 1D tensor; it is required by converters to avoid need to massively rewrite them in that part where constant attributes are queried
            pt_element_type = str(pt_value.type().getElementType())
            ivalue = pt_value.toIValue()
            #print(f'List toIValue: {ivalue}, type of it: {type(ivalue)}')
            is_known_type = pt_element_type in pt_to_ov_type_map
            
            # WA to broken ov.Type
            # Detect integer list and process it with a dedicated method
            # TODO: Fix ov.Type and remove this WA
            #if pt_to_py_type_map[pt_element_type] == 'int':
            #    self.as_constant_list_of_ints(ovshape = PartialShape([len(ivalue)]), ivalue)
            # End of WA to broken ov.Type

            if is_known_type:
                ovtype = pt_to_ov_type_map[pt_element_type]
                #print(f'ovtype = {ovtype}, pt_element_type = {pt_element_type}, OVType.i32 = {OVType.i32}, {OVType.f32}')
                ovshape = PartialShape([len(ivalue)])
                ov_const = op.Constant(ovtype, ovshape.get_shape(), ivalue)
                return ov_const.outputs()

        def input_is_none (self, index):
            if index >= len(self.inputs()) or self._raw_input(index) is None:
                return True
            else:
                r_input = self._raw_input(index)
                return str(r_input.type()) in ['torch.NoneType', 'NoneType']


        def debug (self):
            print(f'DEBUG CALLED FOR {self._raw_output(0)}')
            #self.pt_module.print()

except ImportError as err:
    raise ImportError("OpenVINO Pytorch frontend is not available, please make sure the frontend is built."
                      "{}".format(err))
