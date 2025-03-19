# type: ignore
"""
Factory functions for ops added to openvino opset15.
"""
from functools import partial
from __future__ import annotations
from openvino.opset1.ops import convert_like
from openvino._pyopenvino import Node
from openvino._pyopenvino import Type
from openvino.utils.decorators import binary_op
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_nodes
import functools
import numpy as np
import openvino._pyopenvino
import openvino.utils.decorators
import typing
__all__ = ['Node', 'NodeInput', 'Type', 'as_nodes', 'binary_op', 'bitwise_left_shift', 'bitwise_right_shift', 'col2im', 'constant', 'convert_like', 'embedding_bag_offsets', 'embedding_bag_packed', 'nameable_op', 'np', 'partial', 'roi_align_rotated', 'scatter_nd_update', 'search_sorted', 'slice_scatter', 'squeeze', 'stft', 'string_tensor_pack', 'string_tensor_unpack']
def bitwise_left_shift(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs BitwiseLeftShift operation on input nodes element-wise.
    
        :param arg0: Node with data to be shifted.
        :param arg1: Node with number of shifts.
        :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors.
                               Defaults to “NUMPY”.
    
        :return: The new node performing BitwiseLeftShift operation.
        
    """
def bitwise_right_shift(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs BitwiseRightShift operation on input nodes element-wise.
    
        :param arg0: Tensor with data to be shifted.
        :param arg1: Tensor with number of shifts.
        :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors.
                               Defaults to “NUMPY”.
    
        :return: The new node performing BitwiseRightShift operation.
        
    """
def col2im(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform data movement operation which combines sliding blocks into an image tensor.
    
        :param  data:                The node providing input data.
        :param  output_size:         Shape of the spatial dimensions of the output image.
        :param  kernel_size:         Size of the sliding blocks.
        :param  strides:             Stride on the sliding blocks in the input spatial dimensions. Defaults to [1, 1].
        :param  dilations:           The dilation of filter elements (distance between elements). Defaults to [1, 1].
        :param  pads_begin:          The number of pixels added at the beginning along each axis. Defaults to [0, 0].
        :param  pads_end:            The number of pixels added at the end along each axis. Defaults to [0, 0].
        :param  name:                The optional name for the created output node.
    
        :return:   The new node performing Col2Im operation.
        
    """
def embedding_bag_offsets(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs sums or means of bags of embeddings without the intermediate embeddings.
    
        :param emb_table: Tensor containing the embedding lookup table.
        :param indices: 1D Tensor with indices.
        :param offsets: 1D Tensor containing the starting index positions of each bag in indices.
        :param per_sample_weights: Tensor with weights for each sample.
        :param default_index: Scalar containing default index in embedding table to fill empty bags.
                              If unset or set to -1, empty bags will be filled with 0.
                              Reverse indexing using negative indices is not supported.
        :param reduction: String to select algorithm used to perform reduction of elements in bag.
        :param name: Optional name for output node.
        :return: The new node performing EmbeddingBagOffsets operation.
        
    """
def embedding_bag_packed(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs sums or means of "bags" of embeddings, without the intermediate embeddings.
    
        :param emb_table: Tensor containing the embedding lookup table.
        :param indices: 2D Tensor of shape [batch, indices_per_bag] with indices.
        :param per_sample_weights: Tensor of weights to be multiplied with embedding table with same shape as indices.
        :param reduction: Operator to perform reduction of elements in bag.
        :param name: Optional name for output node.
        :return: The new node performing EmbeddingBagPacked operation.
        
    """
def roi_align_rotated(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs ROIAlignRotated operation.
    
        :param data: Input data.
        :param rois: RoIs (Regions of Interest) to pool over.
        :param batch_indices: Tensor with each element denoting the index of
                              the corresponding image in the batch.
        :param pooled_h: Height of the ROI output feature map.
        :param pooled_w: Width of the ROI output feature map.
        :param sampling_ratio: Number of bins over height and width to use to calculate
                               each output feature map element.
        :param spatial_scale: Multiplicative spatial scale factor to translate ROI coordinates.
        :param clockwise_mode:  If true, rotation angle is interpreted as clockwise,
                                otherwise as counterclockwise
        :param name: The optional name for the output node
    
        :return: The new node which performs ROIAlignRotated
        
    """
def scatter_nd_update(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs ScatterNDUpdate.
    
        :param data: Node input representing the tensor to be updated.
        :param indices: Node input representing the indices at which updates will be applied.
        :param updates: Node input representing the updates to be applied.
        :param reduction: The type of operation to perform on the inputs. One of "none", "sum",
                          "sub", "prod", "min", "max".
        :param name: Optional name for the output node.
        :return: New node performing the ScatterNDUpdate.
        
    """
def search_sorted(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which generates SearchSorted operation.
    
        :param sorted_sequence: The node providing sorted sequence to search in.
        :param values: The node providing searched values.
        :param right_mode: If set to False, return the first suitable index that is found for given value.
                           If set to True, return the last such index. Defaults to False.
        :param name: The optional name for the created output node.
        :return: The new node performing SearchSorted operation.
        
    """
def slice_scatter(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which generates SliceScatter operation.
    
        :param  data: The node providing input data.
        :param  updates: The node providing updates data.
        :param  start: The node providing start indices (inclusively).
        :param  stop: The node providing stop indices (exclusively).
        :param  step: The node providing step values.
        :param  axes: The optional node providing axes to slice, default [0, 1, ..., len(start)-1].
        :param  name: The optional name for the created output node.
        :return: The new node performing SliceScatter operation.
        
    """
def squeeze(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform squeeze operation on input tensor.
    
        :param data: The node with data tensor.
        :param axes: Optional list of integers, indicating the dimensions to squeeze.
                      Negative indices are supported. One of: input node or array.
        :param allow_axis_skip: If true, shape inference results in a dynamic rank, when
                      selected axis has value 1 in its dynamic range. Used only if axes input
                      is given. Defaults to false.
        :param name: Optional new name for output node.
        :return: The new node performing a squeeze operation on input tensor.
    
        Remove single-dimensional entries from the shape of a tensor.
        Takes an optional parameter `axes` with a list of axes to squeeze.
        If `axes` is not provided, all the single dimensions will be removed from the shape.
    
        For example:
    
           Inputs: tensor with shape [1, 2, 1, 3, 1, 1], axes=[2, 4]
    
           Result: tensor with shape [1, 2, 3, 1]
        
    """
def stft(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which generates STFT operation.
    
        :param  data: The node providing input data.
        :param  window: The node providing window data.
        :param  frame_size: The node with scalar value representing the size of Fourier Transform.
        :param  frame_step: The distance (number of samples) between successive window frames.
        :param  transpose_frames: Flag to set output shape layout. If true the `frames` dimension is at out_shape[2],
                                  otherwise it is at out_shape[1].
        :param  name: The optional name for the created output node.
        :return: The new node performing STFT operation.
        
    """
def string_tensor_pack(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform an operation which packs a concatenated batch of strings into a batched string tensor.
    
        :param begins: ND tensor of non-negative integer numbers containing indices of each string's beginnings.
        :param ends: ND tensor of non-negative integer numbers containing indices of each string's endings.
        :param symbols: 1D tensor of concatenated strings data encoded in utf-8 bytes.
    
        :return: The new node performing StringTensorPack operation.
        
    """
def string_tensor_unpack(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform an operation which unpacks a batch of strings into three tensors.
    
        :param data: The node providing input data.
    
        :return: The new node performing StringTensorUnpack operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset15: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset15')
constant: openvino.utils.decorators.MultiMethod  # value = <openvino.utils.decorators.MultiMethod object>
