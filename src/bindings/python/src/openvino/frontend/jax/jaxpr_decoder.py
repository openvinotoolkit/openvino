# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import jax.core
from openvino.frontend.jax.py_jax_frontend import _FrontEndJaxDecoder as Decoder
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape
from openvino.frontend.jax.utils import jax_to_ov_type_map, jax_array_to_ov_const

import jax

from typing import List
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class JaxprPythonDecoder (Decoder):
    '''
    The jaxpr decoder uses Jaxpr to get graph information from a jax module.
    It takes use of the following parts.
    
    - `ClosedJaxpr`: the jaxpr object that contains the jaxpr and literals.
        - `Jaxpr`: the jaxpr object that contains the invars, outvars, and eqns.
            - `JaxEqns`: A list of jaxpr equations, which contains the information of the operation.
                - `Primitive`: the operation that is used in the equation.
                - `invars`: the input variables of the equation.
                    - `aval`: the abstract value.
                - `outvars`: the output variables of the equation.
                    - `aval`: the abstract value.
            - `invars`: the input variables of the equation.
                - `aval`: the abstract value.
            - `outvars`: the output variables of the equation.
                - `aval`: the abstract value.
            - `constvars`: the constant variables.
                - `aval`: the abstract value.
        - `Literal`: the literal object that contains the value of the constants.
    '''
    
    def __init__(self, jaxpr, name=None, literals=None):
        '''
        Inputs: 
            - jaxpr: for users, `ClosedJaxpr` is expected here. See https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L197
            - name: the name for the model.
            - literals: the literals (constants) that are used in the model.
        '''
        Decoder.__init__(self)
        
        if isinstance(jaxpr, (jax.core.JaxprEqn, jax.core.Jaxpr, jax.core.Var)):
            self.jaxpr = jaxpr
        elif isinstance(jaxpr, jax.core.ClosedJaxpr):
            # Take the `Jaxpr` from `ClosedJaxpr`, see https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L85
            self.jaxpr = jaxpr.jaxpr
            # Literal should be a `Jax.core.Var`, see https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L85
            self.literals = jaxpr.literals
        else:
            raise ValueError(f"Unexpected type of jaxpr: {type(jaxpr)}")
        self.name = name
        if self.name is None:
            self.name = "jax_module"
        if literals is not None:
            self.literals = literals
        
        # TODO: this implementation may lead to memory increasing. Any better solution?
        self.m_decoders = []
        
    def inputs(self) -> List[int]:
        if isinstance(self.jaxpr, jax.core.Var):
            return []
        else:
            
            return [id(v) for v in self.jaxpr.invars]
    
    def input(self, idx: int) -> int:
        if isinstance(self.jaxpr, jax.core.Var):
            raise IndexError("The jaxpr is a constant, which does not have input.")
        else:
            return id(self.jaxpr.invars[idx])
    
    def get_input_shape(self, index):
        if isinstance(self.jaxpr, jax.core.Var):
            raise IndexError("The jaxpr is a constant, which does not have input shape.")
        else:
            return PartialShape(self.jaxpr.invars[index].aval.shape)
    
    def get_input_signature_name(self, index) -> str:
        return "jaxpr_invar_" + str(index)
    
    def get_input_type(self, index) -> OVType:
        if isinstance(self.jaxpr, jax.core.Var):
            raise IndexError("The jaxpr is a constant, which does not have input type.")
        else:
            return self.get_type_for_value(self.jaxpr.invars[index])
    
    def get_output_type(self, index) -> OVType:
        if isinstance(self.jaxpr, jax.core.Var):
            return self.get_type_for_value(self.jaxpr)
        else:
            return self.get_type_for_value(self.jaxpr.outvars[index])
        
    def get_output_name(self, index) -> str:
        return "jaxpr_outvar_" + str(index)
    
    def get_output_shape(self, index):
        if isinstance(self.jaxpr, jax.core.Var):
            return PartialShape(self.jaxpr.aval.shape)
        else:
            return PartialShape(self.jaxpr.outvars[index].aval.shape)
    
    def visit_subgraph(self, node_visitor) -> None:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            return
        for idx, node in enumerate(self.jaxpr.constvars):
            decoder = JaxprPythonDecoder(node, name=self.name + "/" + f"const({id(node)})", literals=self.literals[idx])
            self.m_decoders.append(decoder)
            node_visitor(decoder)
        # Visit every `JaxEqn` in the jaxpr, see https://github.com/google/jax/blob/jaxlib-v0.4.29/jax/_src/core.py#L285
        for node in self.jaxpr.eqns:
            decoder = JaxprPythonDecoder(node, name=self.name + "/" + node.primitive.name)
            self.m_decoders.append(decoder)
            node_visitor(decoder)
            
    def get_op_type(self) -> str:
        if isinstance(self.jaxpr, jax.core.JaxprEqn):
            return self.jaxpr.primitive.name
        elif isinstance(self.jaxpr, jax.core.Var):
            return "constant"
        else:
            return "root"
        
    def mark_node(self, node):
        name = self.get_op_type()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        node.set_friendly_name(self.name + "/" + name)
        return node
    
    def outputs(self) -> List[int]:
        if isinstance(self.jaxpr, jax.core.Var):
            return [id(self.jaxpr)]
        else:
            return [id(v) for v in self.jaxpr.outvars]
    
    def output(self, idx: int) -> int:
        if isinstance(self.jaxpr, jax.core.Var):
            return id(self.jaxpr)
        else:
            return id(self.jaxpr.outvars[idx])
    
    def num_inputs(self) -> int:
        if isinstance(self.jaxpr, jax.core.Var):
            return 0
        else:
            return len(self.jaxpr.invars)
    
    def num_outputs(self) -> int:
        if isinstance(self.jaxpr, jax.core.Var):
            return 1
        else:
            return len(self.jaxpr.outvars)
    
    def as_constant(self):
        if self.get_op_type() == 'constant':
            value = self.literals
            # TODO: dig out how to share the memory.
            # Currently, using shared_memory will raise `ValueError: array is not writeable``
            ov_const = jax_array_to_ov_const(value, shared_memory=False)
            return ov_const.outputs()
        else:
            raise ValueError("This is not a constant node so it cannot be converted to a constant.")

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