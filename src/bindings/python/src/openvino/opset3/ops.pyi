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
__all__ = ['Constant', 'Node', 'NodeFactory', 'NodeInput', 'NumericData', 'NumericType', 'Parameter', 'ScalarData', 'Shape', 'TensorShape', 'as_node', 'as_nodes', 'assert_list_of_ints', 'assign', 'binary_op', 'broadcast', 'bucketize', 'check_valid_attributes', 'cum_sum', 'embedding_bag_offsets_sum', 'embedding_bag_packed_sum', 'embedding_segments_sum', 'extract_image_patches', 'get_dtype', 'get_element_type', 'get_element_type_str', 'gru_cell', 'is_non_negative_value', 'is_positive_value', 'make_constant_node', 'nameable_op', 'non_max_suppression', 'non_zero', 'np', 'partial', 'read_value', 'rnn_cell', 'roi_align', 'scatter_elements_update', 'scatter_update', 'shape_of', 'shuffle_channels', 'topk', 'unary_op']
def assign(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the Assign operation.
    
        :param new_value:    Node producing a value to be assigned to a variable.
        :param variable_id:  Id of a variable to be updated.
        :param name:         Optional name for output node.
        :return: Assign node
        
    """
def broadcast(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Create a node which broadcasts the input node's values along specified axes to a desired shape.
    
        :param data: The node with input tensor data.
        :param target_shape: The node with a new shape we want to broadcast tensor to.
        :param axes_mapping: The node with a axis positions (0-based) in the result
                               that are being broadcast.
        :param broadcast_spec: The type of broadcasting that specifies mapping of input tensor axes
                               to output shape axes. Range of values: NUMPY, EXPLICIT, BIDIRECTIONAL.
        :param name: Optional new name for output node.
        :return: New node with broadcast shape.
        
    """
def bucketize(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the Bucketize operation.
    
        :param data:              Input data to bucketize
        :param buckets:           1-D of sorted unique boundaries for buckets
        :param output_type:       Output tensor type, "i64" or "i32", defaults to i64
        :param with_right_bound:  indicates whether bucket includes the right or left
                                  edge of interval. default true = includes right edge
        :param name:              Optional name for output node.
        :return: Bucketize node
        
    """
def cum_sum(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Construct a cumulative summation operation.
    
        :param arg: The tensor to be summed.
        :param axis: zero dimension tensor specifying axis position along which sum will be performed.
        :param exclusive: if set to true, the top element is not included
        :param reverse: if set to true, will perform the sums in reverse direction
        :return: New node performing the operation
        
    """
def embedding_bag_offsets_sum(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs sums of bags of embeddings without the intermediate embeddings.
    
        :param emb_table: Tensor containing the embedding lookup table.
        :param indices: Tensor with indices.
        :param offsets: Tensor containing the starting index positions of each bag in indices.
        :param per_sample_weights: Tensor with weights for each sample.
        :param default_index: Scalar containing default index in embedding table to fill empty bags.
        :param name: Optional name for output node.
        :return: The new node which performs EmbeddingBagOffsetsSum
        
    """
def embedding_bag_packed_sum(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return an EmbeddingBagPackedSum node.
    
        EmbeddingSegmentsSum constructs an output tensor by replacing every index in a given
        input tensor with a row (from the weights matrix) at that index
    
        :param emb_table: Tensor containing the embedding lookup table.
        :param indices: Tensor with indices.
        :param per_sample_weights: Weights to be multiplied with embedding table.
        :param name: Optional name for output node.
        :return: EmbeddingBagPackedSum node
        
    """
def embedding_segments_sum(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return an EmbeddingSegmentsSum node.
    
        EmbeddingSegmentsSum constructs an output tensor by replacing every index in a given
        input tensor with a row (from the weights matrix) at that index
    
        :param emb_table: Tensor containing the embedding lookup table.
        :param indices: Tensor with indices.
        :param segment_ids: Tensor with indices into the output Tensor
        :param num_segments: Tensor with number of segments.
        :param default_index: Scalar containing default index in embedding table to fill empty bags.
        :param per_sample_weights: Weights to be multiplied with embedding table.
        :param name: Optional name for output node.
        :return: EmbeddingSegmentsSum node
        
    """
def extract_image_patches(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the ExtractImagePatches operation.
    
        :param image:     4-D Input data to extract image patches.
        :param sizes:     Patch size in the format of [size_rows, size_cols].
        :param strides:   Patch movement stride in the format of [stride_rows, stride_cols]
        :param rates:     Element seleciton rate for creating a patch.
        :param auto_pad:  Padding type.
        :param name:      Optional name for output node.
        :return: ExtractImagePatches node
        
    """
def gru_cell(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform GRUCell operation on the tensor from input node.
    
        GRUCell represents a single GRU Cell that computes the output
        using the formula described in the paper: https://arxiv.org/abs/1406.1078
    
        Note this class represents only single *cell* and not whole *layer*.
    
        :param X:                       The input tensor with shape: [batch_size, input_size].
        :param initial_hidden_state:    The hidden state tensor at current time step with shape:
                                        [batch_size, hidden_size].
        :param W:                       The weights for matrix multiplication, gate order: zrh.
                                        Shape: [3*hidden_size, input_size].
        :param R:                       The recurrence weights for matrix multiplication.
                                        Shape: [3*hidden_size, hidden_size].
        :param B:                       The sum of biases (weight and recurrence).
                                        For linear_before_reset set True the shape is [4*hidden_size].
                                        Otherwise the shape is [3*hidden_size].
        :param hidden_size:             The number of hidden units for recurrent cell.
                                        Specifies hidden state size.
        :param activations:             The vector of activation functions used inside recurrent cell.
        :param activation_alpha:        The vector of alpha parameters for activation functions in
                                        order respective to activation list.
        :param activation_beta:         The vector of beta parameters for activation functions in order
                                        respective to activation list.
        :param clip:                    The value defining clipping range [-clip, clip] on input of
                                        activation functions.
        :param linear_before_reset:     Flag denotes if the layer behaves according to the modification
                                        of GRUCell described in the formula in the ONNX documentation.
        :param name:                    Optional output node name.
        :return:   The new node performing a GRUCell operation on tensor from input node.
        
    """
def non_max_suppression(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs NonMaxSuppression.
    
        :param boxes: Tensor with box coordinates.
        :param scores: Tensor with box scores.
        :param max_output_boxes_per_class: Tensor Specifying maximum number of boxes
                                            to be selected per class.
        :param iou_threshold: Tensor specifying intersection over union threshold
        :param score_threshold: Tensor specifying minimum score to consider box for the processing.
        :param box_encoding: Format of boxes data encoding.
        :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                       boxes across batches or not.
        :param output_type: Output element type.
        :return: The new node which performs NonMaxSuppression
        
    """
def non_zero(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return the indices of the elements that are non-zero.
    
        :param data: Input data.
        :param output_type: Output tensor type.
    
        :return: The new node which performs NonZero
        
    """
def read_value(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the Assign operation.
    
        :param init_value:   Node producing a value to be returned instead of an unassigned variable.
        :param variable_id:  Id of a variable to be read.
        :param name:         Optional name for output node.
        :return: ReadValue node
        
    """
def rnn_cell(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform RNNCell operation on tensor from input node.
    
        It follows notation and equations defined as in ONNX standard:
        https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN
    
        Note this class represents only single *cell* and not whole RNN *layer*.
    
        :param X:                       The input tensor with shape: [batch_size, input_size].
        :param initial_hidden_state:    The hidden state tensor at current time step with shape:
                                        [batch_size, hidden_size].
        :param W:                       The weight tensor with shape: [hidden_size, input_size].
        :param R:                       The recurrence weight tensor with shape: [hidden_size,
                                        hidden_size].
        :param B:                       The sum of biases (weight and recurrence) with shape: [hidden_size].
        :param hidden_size:             The number of hidden units for recurrent cell.
                                        Specifies hidden state size.
        :param activations:             The vector of activation functions used inside recurrent cell.
        :param activation_alpha:        The vector of alpha parameters for activation functions in
                                        order respective to activation list.
        :param activation_beta:         The vector of beta parameters for activation functions in order
                                        respective to activation list.
        :param clip:                    The value defining clipping range [-clip, clip] on input of
                                        activation functions.
        :param name:                    Optional output node name.
        :return:   The new node performing a RNNCell operation on tensor from input node.
        
    """
def roi_align(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs ROIAlign.
    
        :param data: Input data.
        :param rois: RoIs (Regions of Interest) to pool over.
        :param batch_indices: Tensor with each element denoting the index of
                              the corresponding image in the batch.
        :param pooled_h: Height of the ROI output feature map.
        :param pooled_w: Width of the ROI output feature map.
        :param sampling_ratio: Number of bins over height and width to use to calculate
                               each output feature map element.
        :param spatial_scale: Multiplicative spatial scale factor to translate ROI coordinates.
        :param mode: Method to perform pooling to produce output feature map elements.
    
        :return: The new node which performs ROIAlign
        
    """
def scatter_elements_update(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a ScatterElementsUpdate operation.
    
        :param data:    The input tensor to be updated.
        :param indices: The tensor with indexes which will be updated.
        :param updates: The tensor with update values.
        :param axis:    The axis for scatter.
        :return: ScatterElementsUpdate node
    
        ScatterElementsUpdate creates a copy of the first input tensor with updated elements
        specified with second and third input tensors.
    
        For each entry in `updates`, the target index in `data` is obtained by combining
        the corresponding entry in `indices` with the index of the entry itself: the
        index-value for dimension equal to `axis` is obtained from the value of the
        corresponding entry in `indices` and the index-value for dimension not equal
        to `axis` is obtained from the index of the entry itself.
    
        
    """
def scatter_update(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a ScatterUpdate operation.
    
        ScatterUpdate sets new values to slices from data addressed by indices.
    
        :param data:    The input tensor to be updated.
        :param indices: The tensor with indexes which will be updated.
        :param updates: The tensor with update values.
        :param axis:    The axis at which elements will be updated.
        :return: ScatterUpdate node
        
    """
def shape_of(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a tensor containing the shape of its input data.
    
        :param data: The tensor containing the input data.
        :param output_type: Output element type.
        :return: ShapeOf node
        
    """
def shuffle_channels(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform permutation on data in the channel dimension of the input tensor.
    
        :param data: The node with input tensor.
        :param axis: Channel dimension index in the data tensor.
                     A negative value means that the index should be calculated
                     from the back of the input data shape.
        :param group: The channel dimension specified by the axis parameter
                     should be split into this number of groups.
        :param name: Optional output node name.
        :return: The new node performing a permutation on data in the channel dimension
                 of the input tensor.
    
        The operation is the equivalent with the following transformation of the input tensor
        `data` of shape [N, C, H, W]:
    
        `data_reshaped` = reshape(`data`, [N, group, C / group, H * W])
    
        `data_transposed` = transpose(`data_reshaped`, [0, 2, 1, 3])
    
        `output` = reshape(`data_transposed`, [N, C, H, W])
    
        For example:
    
        .. code-block:: python
    
            Inputs: tensor of shape [1, 6, 2, 2]
    
                    data = [[[[ 0.,  1.], [ 2.,  3.]],
                             [[ 4.,  5.], [ 6.,  7.]],
                             [[ 8.,  9.], [10., 11.]],
                             [[12., 13.], [14., 15.]],
                             [[16., 17.], [18., 19.]],
                             [[20., 21.], [22., 23.]]]]
    
                    axis = 1
                    groups = 3
    
            Output: tensor of shape [1, 6, 2, 2]
    
                    output = [[[[ 0.,  1.], [ 2.,  3.]],
                               [[ 8.,  9.], [10., 11.]],
                               [[16., 17.], [18., 19.]],
                               [[ 4.,  5.], [ 6.,  7.]],
                               [[12., 13.], [14., 15.]],
                               [[20., 21.], [22., 23.]]]]
        
    """
def topk(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs TopK.
    
        :param data: Input data.
        :param k: K.
        :param axis: TopK Axis.
        :param mode: Compute TopK largest ('max') or smallest ('min')
        :param sort: Order of output elements (sort by: 'none', 'index' or 'value')
        :param index_element_type: Type of output tensor with indices.
        :return: The new node which performs TopK (both indices and values)
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset3: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset3')
