# type: ignore
from __future__ import annotations
from functools import partial
from openvino._pyopenvino import Node
from openvino.utils.decorators import unary_op
from openvino.utils.node_factory import _get_node_factory
import functools
import openvino._pyopenvino
import typing
"""
Factory functions for ops added to openvino opset17.
"""
__all__: list[str] = ['Node', 'NodeInput', 'erfinv', 'partial', 'unary_op']
def erfinv(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which calculates the inverse error function element-wise on the input tensor.
    
        :param node: The node providing data for the operation. Must be of floating-point type.
        :param name: The optional name for the new output node.
        :return: The new node performing the element-wise ErfInv operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset17: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset17')
