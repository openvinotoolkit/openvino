# type: ignore
"""
Factory functions for ops added to openvino opset16.
"""
from functools import partial
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_nodes
import functools
import openvino._pyopenvino
import typing
__all__ = ['Node', 'NodeInput', 'as_nodes', 'identity', 'istft', 'nameable_op', 'partial', 'segment_max']
def identity(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Identity operation is used as a placeholder. It creates a copy of the input to forward to the output.
    
        :param data: Tensor with data.
    
        :return: The new node performing Identity operation.
        
    """
def istft(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which generates ISTFT operation.
    
        :param  data: The node providing input data.
        :param  window: The node providing window data.
        :param  frame_size: The node with scalar value representing the size of Fourier Transform.
        :param  frame_step: The distance (number of samples) between successive window frames.
        :param  center: Flag signaling if the signal input has been padded before STFT.
        :param  normalized: Flag signaling if the STFT result has been normalized.
        :param  signal_length: The optional node with length of the original signal.
        :param  name: The optional name for the created output node.
        :return: The new node performing ISTFT operation.
        
    """
def segment_max(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    The SegmentMax operation finds the maximum value in each specified segment of the input tensor.
    
        :param data: ND tensor of type T, the numerical data on which SegmentMax operation will be performed.
        :param segment_ids: 1D Tensor of sorted non-negative numbers, representing the segments.
        :param num_segments: An optional scalar value representing the segments count. If not provided, it is inferred from segment_ids.
        :param fill_mode: Responsible for the value assigned to segments which are empty. Can be "ZERO" or "LOWEST".
        :param name: Optional name for the node.
    
        :return: The new node performing SegmentMax operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset16: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset16')
