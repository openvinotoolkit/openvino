# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import jax.core
from openvino.frontend.jax.py_jax_frontend import _FrontEndJaxDecoder as Decoder
# from openvino.frontend.jax.py_jax_frontend import _Type as DecoderType
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape
from openvino.frontend.jax.utils import jax_to_ov_type_map

import jax

from typing import List
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class JaxprPythonDecoder (Decoder):
    def __init__(self, jaxpr, name=None):
        Decoder.__init__(self)
        
        # TODO: the `self.jaxpr` here is possible to be a jaxpr or a jax_eqn. Maybe we need a better design.
        if isinstance(jaxpr, (jax.core.Jaxpr, jax.core.JaxprEqn)):
            self.jaxpr = jaxpr
        elif isinstance(jaxpr, jax.core.ClosedJaxpr):
            self.jaxpr = jaxpr.jaxpr
        else:
            raise ValueError(f"Unexpected type of jaxpr: {type(jaxpr)}")
        self.name = name
        if self.name is None:
            self.name = "jax_module"
        
        # TODO: this implementation may lead to memory increasing. Any better solution?
        self.m_decoders = []
        
    def inputs(self) -> List[int]:
        return [id(v) for v in self.jaxpr.invars]
    
    def input(self, idx: int) -> int:
        return id(self.jaxpr.invars[idx])
    
    def get_attribute(self, name):
        return OVAny(None)
    
    def get_input_debug_name(self, index) -> str:
        return "jaxpr_invar_" + str(index)
    
    def get_input_shape(self, index):
        return PartialShape(self.jaxpr.invars[index].aval.shape)
    
    def get_input_signature_name(self, index) -> str:
        # TODO: add a real signature name here
        return self.get_input_debug_name(index)
    
    def get_input_type(self, index) -> OVType:
        return self.get_type_for_value(self.jaxpr.invars[index])
    
    def get_named_input(self, name):
        # TODO: check again if there's named input in jaxpr
        raise NotImplementedError("Currently named input is not expected in jax.")
        
    def get_output_debug_name(self, index) -> str:
        return "jaxpr_outvar_" + str(index)
    
    def get_output_type(self, index) -> OVType:
        return self.get_type_for_value(self.jaxpr.outvars[index])
    
    def get_output_shape(self, index):
        return PartialShape(self.jaxpr.outvars[index].aval.shape)
    
    def decoder_type_name(self) -> str:
        return "jaxpr"
    
    def visit_subgraph(self, node_visitor) -> None:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            return
        for node in self.jaxpr.eqns:
            decoder = JaxprPythonDecoder(node, name=self.name + "/" + node.primitive.name)
            self.m_decoders.append(decoder)
            node_visitor(decoder)
            
    def get_op_type(self) -> str:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            return self.jaxpr.primitive.name
        else:
            return "root"
        
    def get_schema(self) -> str:
        # TODO: this is not really the semantic of schema in jaxpr. Need to check again.
        return str(self.jaxpr)
    
    def get_subgraph_decoder(self, index: int) -> Decoder:
        raise NotImplementedError("Haven't figured out when it should be used.")
    
    def get_subgraph_size(self) -> int:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            return 0
        else:
            # TODO: check again if there's no subgraph in jaxpr
            return 1
        
    def mark_node(self, node):
        name = self.get_op_type()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        node.set_friendly_name(self.name + "/" + name)
        return node
    
    def outputs(self) -> List[int]:
        return [id(v) for v in self.jaxpr.outvars]
    
    def output(self, idx: int) -> int:
        return id(self.jaxpr.outvars[idx])
    
    def num_inputs(self) -> int:
        return len(self.jaxpr.invars)
    
    def num_outputs(self) -> int:
        return len(self.jaxpr.outvars)
    
    def as_constant(self):
        # This may not be necessary in jax frontend but needs further check.
        raise NotImplementedError("Not implemented yet.")
    
    def as_string(self):
        return str(self.jaxpr)

    def input_is_none(self, index) -> bool:
        return self.jaxpr.invars[index] is None
     
    @staticmethod
    def get_type_for_value(value):
        if isinstance(value, (jax.core.Var, jax.core.Literal)):
            for k, v in jax_to_ov_type_map.items():
                if isinstance(value.aval.dtype, k):
                    return OVAny(v)
        elif isinstance(value, (int, float, bool)):
            return OVAny(jax_to_ov_type_map[type(value)])
        else:
            raise NotImplementedError(f"dtype for {value} of type {type(value)} has not been supported yet.")