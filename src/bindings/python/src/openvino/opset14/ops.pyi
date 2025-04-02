# type: ignore
"""
Factory functions for ops added to openvino opset14.
"""
from functools import partial
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino._pyopenvino import Type
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
import functools
import openvino._pyopenvino
import typing
__all__ = ['Node', 'NodeInput', 'TensorShape', 'Type', 'as_node', 'as_nodes', 'avg_pool', 'convert_promote_types', 'inverse', 'max_pool', 'nameable_op', 'partial']
def avg_pool(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return average pooling node.
    
        :param data_batch:      The input node providing data.
        :param strides:         The window movement strides.
        :param pads_begin:      The number of pixels to add at the beginning along each axis.
        :param pads_end:        The number of pixels to add at the end along each axis.
        :param kernel_shape:    The pooling window shape.
        :param exclude_pad:     Whether or not to include zero padding in average computations.
        :param rounding_type:   Determines used rounding schema when computing output shape. Acceptable
                                values are: ['floor', 'ceil', 'ceil_torch']. Defaults to 'floor'.
        :param auto_pad:        Determines how the padding is calculated. Acceptable values:
                                [None, 'same_upper', 'same_lower', 'valid']. Defaults to None.
        :param name:            Optional name for the new output node.
    
        :return: New node with AvgPool operation applied on its data.
        
    """
def convert_promote_types(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node performing conversion to common type based on promotion rules.
    
        :param left_node: input node with type to be promoted to common one.
        :param right_node: input node with type to be promoted to common one.
        :param promote_unsafe: Bool attribute whether to allow promotions that might result in bit-widening, precision loss and undefined behaviors.
        :param pytorch_scalar_promotion: Bool attribute whether to promote scalar input to type provided by non-scalar input when number format is matching.
        :param u64_integer_promotion_target: Element type attribute to select promotion result when inputs are u64 and signed integer.
        :param name: Optional name for the new output node.
    
        :return: The new node performing ConvertPromoteTypes operation.
        
    """
def inverse(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node with inverse matrices of the input.
    
        :param data: Tensor with matrices to invert. Last two dimensions must be of the same size.
        :param adjoint: Whether to return adjoint instead of inverse matrices. Defaults to false.
        :param name: Optional name for the new output node.
    
        :return: The new node performing Inverse operation.
        
    """
def max_pool(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform max pooling operation and return both values and indices of the selected elements.
    
        :param  data:                The node providing input data.
        :param  strides:             The distance (in pixels) to slide the filter on the feature map
                                     over the axes.
        :param  dilations:           The dilation of filter elements(distance between elements).
        :param  pads_begin:          The number of pixels to add at the beginning along each axis.
        :param  pads_end:            The number of pixels to add at the end along each axis.
        :param  kernel_shape:        The pooling operation kernel shape.
        :param  rounding_type:       Determines used rounding schema when computing output shape.
                                     Acceptable values are: ['floor', 'ceil', 'ceil_torch']. Defaults to 'floor'.
        :param  auto_pad:            Determines how the padding is calculated. Acceptable values:
                                     [None, 'same_upper', 'same_lower', 'valid']. Defaults to None.
        :param  index_element_type:  The data type used for the indices output of this operator.
                                     Defaults to i64.
        :param  axis:                The first dimension in the data shape used to determine the maximum
                                     returned index value. The value is the product of all dimensions
                                     starting at the provided axis. Defaults to 0.
        :param  name:                The optional name for the created output node.
    
        :return:   The new node performing max pooling operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset14: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset14')
