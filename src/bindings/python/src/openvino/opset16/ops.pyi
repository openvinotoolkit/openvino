# type: ignore
from __future__ import annotations
from builtins import list as TensorShape
from functools import partial
from openvino._pyopenvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
import functools
import openvino._pyopenvino
import typing
"""
Factory functions for ops added to openvino opset16.
"""
__all__: list[str] = ['Node', 'NodeInput', 'TensorShape', 'as_node', 'as_nodes', 'avg_pool', 'identity', 'istft', 'nameable_op', 'one_hot', 'partial', 'segment_max', 'sparse_fill_empty_rows']
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
        :param dilations:       The index of the next pixel to select when pooling. If not provided,
                                defaults to [1, 1, ...] (no dilation).
        :param name:            Optional name for the new output node.
    
        :return: New node with AvgPool operation applied on its data.
        
    """
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
def one_hot(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Create node performing one-hot encoding on input data.
    
        :param indices: Input tensor of rank N with indices of any supported integer data type.
        :param depth: Scalar of any supported integer type that specifies number of classes and
                      the size of one-hot dimension.
        :param on_value: Scalar of any type that is the value that the locations
                         in output tensor represented by indices in input take.
        :param off_value: Scalar of any type that is the value that the locations not represented
                          by indices in input take.
        :param axis: New axis position in the output shape to fill with one-hot values.
        :param negative_indices_mode: Controls how negative indices are handled. Can be 'ignore_negative'
                                      (negative indices are ignored and filled with off_value) or
                                      'normalize' (negative indices in range [-depth, -1] are normalized).
                                      If not provided, defaults to 'ignore_negative'.
        :param name: The optional name for new output node.
        :return: New node performing one-hot operation.
        
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
def sparse_fill_empty_rows(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Fills empty rows of an input sparse tensor with a default value.
    
        :param values: 1D tensor containing the values to be inserted at the specified indices.
        :param dense_shape: 1D tensor indicating the shape of the 2D dense tensor.
        :param indices: 2D tensor indicating the positions at which values are placed.
        :param default_value: A scalar value to be inserted into empty rows.
        :param name: Optional name for the node.
    
        :return: The new node performing SparseFillEmptyRows operation with three outputs:
                 [output_indices, output_values, empty_row_indicator]
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset16: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset16')
