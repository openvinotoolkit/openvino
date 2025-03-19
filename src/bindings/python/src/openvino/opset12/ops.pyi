# type: ignore
"""
Factory functions for all ngraph ops.
"""
from functools import partial
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
import functools
import openvino._pyopenvino
import typing
__all__ = ['Node', 'NodeInput', 'as_node', 'as_nodes', 'group_normalization', 'nameable_op', 'pad', 'partial', 'scatter_elements_update']
def group_normalization(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a GroupNormalization operation.
    
        :param data:    The input tensor to be normalized.
        :param scale:   The tensor containing the scale values for each channel.
        :param bias:    The tensor containing the bias values for each channel.
        :param num_groups: Specifies the number of groups that the channel dimension will be divided into.
        :param epsilon: A very small value added to the variance for numerical stability.
                        Ensures that division by zero does not occur for any normalized element.
        :return: GroupNormalization node
        
    """
def pad(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a generic padding operation.
    
        :param arg: The node producing input tensor to be padded.
        :param pads_begin: Number of padding elements to be added before position 0
                           on each axis of arg. Negative values are supported.
        :param pads_end: Number of padding elements to be added after the last element.
                         Negative values are supported.
        :param pad_mode: "constant", "edge", "reflect" or "symmetric"
        :param arg_pad_value: value used for padding if pad_mode is "constant"
        :return: Pad operation node.
        
    """
def scatter_elements_update(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a ScatterElementsUpdate operation.
    
        :param data:    The input tensor to be updated.
        :param indices: The tensor with indexes which will be updated. Negative indices are supported.
        :param updates: The tensor with update values.
        :param axis:    The axis for scatter.
        :param reduction: The type of operation to perform on the inputs. One of "none", "sum",
                          "prod", "min", "max", "mean".
        :param: use_init_val: Controls whether the elements in the data input tensor are used as
                              initial value for reduce operations.
        :return: ScatterElementsUpdate node
    
        ScatterElementsUpdate creates a copy of the first input tensor with updated elements
        specified with second and third input tensors.
    
        For each entry in `updates`, the target index in `data` is obtained by combining
        the corresponding entry in `indices` with the index of the entry itself: the
        index-value for dimension equal to `axis` is obtained from the value of the
        corresponding entry in `indices` and the index-value for dimension not equal
        to `axis` is obtained from the index of the entry itself.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset12: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset12')
