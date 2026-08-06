# type: ignore
from __future__ import annotations
from functools import partial
from openvino._pyopenvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.decorators import unary_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_nodes
import functools
import openvino._pyopenvino
import typing
"""
Factory functions for ops added to openvino opset17.
"""
__all__: list[str] = ['Node', 'NodeInput', 'as_nodes', 'erfinv', 'grouped_matmul', 'nameable_op', 'partial', 'unary_op']
def erfinv(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which calculates the inverse error function element-wise on the input tensor.
    
        :param node: The node providing data for the operation. Must be of floating-point type.
        :param name: The optional name for the new output node.
        :return: The new node performing the element-wise ErfInv operation.
        
    """
def grouped_matmul(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs Grouped Matrix Multiplication for Mixture of Experts (MoE).
    
        Computes multiple matrix multiplications where each group processes a subset of the input
        data. Two input combinations are supported:
    
        - Case 1 (2D x 3D), MoE forward pass, requires ``offsets``:
            - mat_a: (total_tokens, K) - rows partitioned by offsets
            - mat_b: (G, N, K) - per-group weights
            - output: (total_tokens, N)
        - Case 2 (3D x 3D), batched uniform, no offsets:
            - mat_a: (G, M, K) - per-group inputs
            - mat_b: (G, N, K) - per-group weights
            - output: (G, M, N)
    
        :param mat_a: The first input tensor.
        :param mat_b: The second input tensor with per-group weights.
        :param offsets: 1D tensor of cumulative end-offsets of shape (G,) indicating
                        group boundaries. Required for the 2D x 3D case.
        :param name: The optional name for the new output node.
    
        :return: The new node performing GroupedMatMul operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset17: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset17')
