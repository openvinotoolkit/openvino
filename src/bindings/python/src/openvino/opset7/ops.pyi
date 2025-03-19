# type: ignore
"""
Factory functions for all openvino ops.
"""
from functools import partial
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino._pyopenvino import Shape
from openvino._pyopenvino.op import Constant
from openvino._pyopenvino.op import Parameter
from openvino.utils.decorators import binary_op
from openvino.utils.decorators import nameable_op
from openvino.utils.decorators import unary_op
from openvino.utils.input_validation import assert_list_of_ints
from openvino.utils.input_validation import check_valid_attributes
from openvino.utils.input_validation import is_non_negative_value
from openvino.utils.input_validation import is_positive_value
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.node_factory import NodeFactory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
from openvino.utils.types import get_dtype
from openvino.utils.types import get_element_type
from openvino.utils.types import get_element_type_str
from openvino.utils.types import make_constant_node
import functools
import numpy as np
import openvino._pyopenvino
import typing
__all__ = ['Constant', 'Node', 'NodeFactory', 'NodeInput', 'NumericData', 'NumericType', 'Parameter', 'ScalarData', 'Shape', 'TensorShape', 'as_node', 'as_nodes', 'assert_list_of_ints', 'binary_op', 'check_valid_attributes', 'dft', 'einsum', 'gather', 'gelu', 'get_dtype', 'get_element_type', 'get_element_type_str', 'idft', 'is_non_negative_value', 'is_positive_value', 'make_constant_node', 'nameable_op', 'np', 'partial', 'roll', 'unary_op']
def dft(data: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], axes: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], signal_size: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray, NoneType] = None, name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Return a node which performs DFT operation.
    
        :param data: Tensor with transformed data.
        :param axes: Tensor with axes to transform.
        :param signal_size: Tensor specifying signal size with respect to axes from the input 'axes'.
        :param name: Optional output node name.
        :return: The new node which performs DFT operation on the input data tensor.
        
    """
def einsum(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs Einsum operation.
    
        :param inputs: The list of input nodes
        :param equation: Einsum equation
        :param name: Optional output node name.
        :return: The new node performing Einsum operation on the inputs
        
    """
def gather(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs Gather.
    
        :param data:         N-D tensor with data for gathering
        :param indices:      N-D tensor with indices by which data is gathered
        :param axis:         axis along which elements are gathered
        :param batch_dims:   number of batch dimensions
        :param name:         Optional output node name.
        :return:             The new node which performs Gather
        
    """
def gelu(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs Gelu activation function.
    
        :param data: The node with data tensor.
        :param approximation_mode: defines which approximation to use ('tanh' or 'erf')
        :param name: Optional output node name.
        :return: The new node performing a Gelu activation with the input tensor.
        
    """
def idft(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs IDFT operation.
    
        :param data: Tensor with transformed data.
        :param axes: Tensor with axes to transform.
        :param signal_size: Tensor specifying signal size with respect to axes from the input 'axes'.
        :param name: Optional output node name.
        :return: The new node which performs IDFT operation on the input data tensor.
        
    """
def roll(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs Roll operation.
    
        :param data: The node with data tensor.
        :param shift: The node with the tensor with numbers of places by which elements are shifted.
        :param axes: The node with the tensor with axes along which elements are shifted.
        :param name: Optional output node name.
        :return: The new node performing a Roll operation on the input tensor.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset7: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset7')
