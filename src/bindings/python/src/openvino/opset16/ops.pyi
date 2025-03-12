"""
Factory functions for ops added to openvino opset16.
"""
from __future__ import annotations
import functools
from functools import partial
import openvino._pyopenvino
from openvino._pyopenvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_nodes
import typing
__all__ = ['Node', 'NodeInput', 'as_nodes', 'identity', 'nameable_op', 'partial']
def identity(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Identity operation is used as a placeholder. It creates a copy of the input to forward to the output.
    
        :param data: Tensor with data.
    
        :return: The new node performing Identity operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset16: functools.partial  # value = functools.partial(<function _get_node_factory at 0x7fb4604e1d00>, 'opset16')
