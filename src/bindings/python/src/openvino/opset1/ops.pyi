# type: ignore
"""
Factory functions for all openvino ops.
"""
from functools import partial
from __future__ import annotations
from openvino.package_utils import deprecated
from openvino._pyopenvino import Node
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Type
from openvino._pyopenvino.op import Constant
from openvino._pyopenvino.op import Parameter
from openvino._pyopenvino.op import tensor_iterator
from openvino.utils.decorators import binary_op
from openvino.utils.decorators import nameable_op
from openvino.utils.decorators import unary_op
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
from typing import get_args
import functools
import numpy as np
import openvino._pyopenvino
import openvino._pyopenvino.op
import typing
__all__ = ['Constant', 'Node', 'NodeFactory', 'NodeInput', 'NumericData', 'NumericType', 'Parameter', 'PartialShape', 'ScalarData', 'TensorShape', 'Type', 'absolute', 'acos', 'add', 'as_node', 'as_nodes', 'asin', 'atan', 'avg_pool', 'batch_norm_inference', 'binary_convolution', 'binary_op', 'broadcast', 'ceiling', 'check_valid_attributes', 'clamp', 'concat', 'constant', 'convert', 'convert_like', 'convolution', 'convolution_backprop_data', 'cos', 'cosh', 'ctc_greedy_decoder', 'deformable_convolution', 'deformable_psroi_pooling', 'deprecated', 'depth_to_space', 'detection_output', 'divide', 'elu', 'equal', 'erf', 'exp', 'fake_quantize', 'floor', 'floor_mod', 'gather', 'gather_tree', 'get_args', 'get_dtype', 'get_element_type', 'get_element_type_str', 'greater', 'greater_equal', 'grn', 'group_convolution', 'group_convolution_backprop_data', 'hard_sigmoid', 'interpolate', 'is_non_negative_value', 'is_positive_value', 'less', 'less_equal', 'log', 'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'lrn', 'lstm_cell', 'make_constant_node', 'matmul', 'max_pool', 'maximum', 'minimum', 'mod', 'multiply', 'nameable_op', 'negative', 'non_max_suppression', 'normalize_l2', 'not_equal', 'np', 'one_hot', 'pad', 'parameter', 'partial', 'power', 'prelu', 'prior_box', 'prior_box_clustered', 'proposal', 'psroi_pooling', 'range', 'reduce_logical_and', 'reduce_logical_or', 'reduce_max', 'reduce_mean', 'reduce_min', 'reduce_prod', 'reduce_sum', 'region_yolo', 'relu', 'reshape', 'result', 'reverse_sequence', 'select', 'selu', 'shape_of', 'sigmoid', 'sign', 'sin', 'sinh', 'softmax', 'space_to_depth', 'split', 'sqrt', 'squared_difference', 'squeeze', 'strided_slice', 'subtract', 'tan', 'tanh', 'tensor_iterator', 'tile', 'topk', 'transpose', 'unary_op', 'unsqueeze', 'variadic_split']
def absolute(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies f(x) = abs(x) to the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with Abs operation applied on it.
        
    """
def acos(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply inverse cosine function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with arccos operation applied on it.
        
    """
def add(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies f(A,B) = A+B to the input nodes element-wise.
    
        :param left_node: The first input node for add operation.
        :param right_node: The second input node for add operation.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors. Defaults to "NUMPY".
        :param name: The optional name for output new node.
        :return: The node performing element-wise addition.
        
    """
def asin(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply inverse sine function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with arcsin operation applied on it.
        
    """
def atan(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply inverse tangent function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with arctan operation applied on it.
        
    """
def avg_pool(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return average pooling node.
    
        :param data_batch:      The input node providing data.
        :param strides:         The window movement strides.
        :param pads_begin:      The input data optional padding below filled with zeros.
        :param pads_end:        The input data optional padding below filled with zeros.
        :param kernel_shape:    The pooling window shape.
        :param exclude_pad:     Whether or not to include zero padding in average computations.
        :param rounding_type:   Determines used rounding schema when computing output shape. Acceptable
                                values are: ['floor', 'ceil']
        :param auto_pad:        Determines how the padding is calculated. Acceptable values:
                                [None, 'same_upper', 'same_lower', 'valid']
        :param name:            Optional name for the new output node.
    
        :return: New node with AvgPool operation applied on its data.
        
    """
def batch_norm_inference(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform layer normalizes a input tensor by mean and variance with appling scale and offset.
    
        :param data: The input tensor with data for normalization.
        :param gamma: The scalar scaling for normalized value.
        :param beta: The bias added to the scaled normalized value.
        :param mean: The value for mean normalization.
        :param variance: The value for variance normalization.
        :param epsilon: The  number to be added to the variance to avoid division
                        by zero when normalizing a value.
        :param name: The optional name of the output node.
        :return: The new node which performs BatchNormInference.
        
    """
def binary_convolution(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Create node performing convolution with binary weights, binary input and integer output.
    
        :param data: The node providing data batch tensor.
        :param filter: The node providing filters tensor.
        :param strides: The kernel window movement strides.
        :param pads_begin: The number of pixels to add to the beginning along each axis.
        :param pads_end: The number of pixels to add to the end along each axis.
        :param dilations: The distance in width and height between elements (weights) in the filter.
        :param mode: Defines how input tensor 0/1 values and weights 0/1 are interpreted.
        :param pad_value: Floating-point value used to fill pad area.
        :param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
        :param name: The optional new name for output node.
        :return: New node performing binary convolution operation.
        
    """
def broadcast(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Create a node which broadcasts the input node's values along specified axes to a desired shape.
    
        :param data: The node with input tensor data.
        :param target_shape: The node with a new shape we want to broadcast tensor to.
        :param axes_mapping: The node with a axis positions (0-based) in the result
                               that are being broadcast.
        :param mode: The type of broadcasting that specifies mapping of input tensor axes
                               to output shape axes. Range of values: NUMPY, EXPLICIT.
        :param name: Optional new name for output node.
        :return: New node with broadcast shape.
        
    """
def ceiling(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies ceiling to the input node element-wise.
    
        :param node: The node providing data to ceiling operation.
        :param name: Optional name for output node.
        :return: The node performing element-wise ceiling.
        
    """
def clamp(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform clamp element-wise on data from input node.
    
        :param data: Input tensor. One of: input node, array or scalar.
        :param min_value: The lower bound of the <min_value;max_value> range. Scalar value.
        :param max_value: The upper bound of the <min_value;max_value> range. Scalar value.
        :param name: Optional output node name.
        :return: The new node performing a clamp operation on its input data element-wise.
    
        Performs a clipping operation on an input value between a pair of boundary values.
    
        For each element in `data`, if the element's value is lower than `min_value`,
        it will be replaced with `min_value`. If the value is higher than `max_value`,
        it will be replaced by `max_value`.
        Intermediate values of `data` are returned without change.
    
        Clamp uses the following logic:
    
        .. code-block:: python
    
            if data < min_value:
                data=min_value
            elif data > max_value:
                data=max_value
        
    """
def concat(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Concatenate input nodes into single new node along specified axis.
    
        :param nodes: The nodes we want concatenate into single new node.
        :param axis: The axis along which we want to concatenate input nodes.
        :param name: The optional new name for output node.
        :return: Return new node that is a concatenation of input nodes.
        
    """
def constant(*args, **kwargs) -> openvino._pyopenvino.op.Constant:
    """
    Create a Constant node from provided value.
    
        :param value: One of: array of values or scalar to initialize node with.
        :param dtype: The data type of provided data.
        :param name: Optional name for output node.
        :return: The Constant node initialized with provided data.
        
    """
def convert(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which casts input node values to specified type.
    
        :param data: Node which produces the input tensor.
        :param destination_type: Provides the target type for the conversion.
        :param name: Optional name for the output node.
        :return: New node performing the conversion operation.
        
    """
def convert_like(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which casts data node values to the type of another node.
    
        :param data: Node which produces the input tensor
        :param like: Node which provides the target type information for the conversion
        :param name: Optional name for the output node.
        :return: New node performing the conversion operation.
        
    """
def convolution(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node performing batched convolution operation.
    
        :param data: The node providing data batch tensor.
        :param filter: The node providing filters tensor.
        :param strides: The kernel window movement strides.
        :param pads_begin: The number of zero padding elements to add on each axis below 0 coordinate.
        :param pads_end: The number of zero padding elements to add on each axis above max coordinate
        :param dilations: The data batch dilation strides.
        :param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
        :param name: The optional new name for output node.
        :return: New node performing batched convolution operation.
        
    """
def convolution_backprop_data(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Create node performing a batched-convolution backprop data operation.
    
        :param      data:         The node producing data from forward-prop
        :param      filters:      The node producing the filters from forward-prop.
        :param      output_shape: The node producing output delta.
        :param      strides:      The distance (in pixels) to slide the filter on the feature map
                                  over the axes.
        :param      pads_begin:   The number of pixels to add to the beginning along each axis.
        :param      pads_end:     The number of pixels to add to the end along each axis.
        :param      dilations:    The distance in width and height between elements (weights)
                                  in the filter.
        :param      name:         The node name.
    
        :return:   The node object representing ConvolutionBackpropData  operation.
        
    """
def cos(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply cosine function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with cos operation applied on it.
        
    """
def cosh(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply hyperbolic cosine function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with cosh operation applied on it.
        
    """
def ctc_greedy_decoder(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform greedy decoding on the logits given in input (best path).
    
        :param data: Logits on which greedy decoding is performed.
        :param sequence_mask: The tensor with sequence masks for each sequence in the batch.
        :param merge_repeated: The flag for merging repeated labels during the CTC calculation.
        :param name: Optional name for output node.
        :return: The new node performing an CTCGreedyDecoder operation on input tensor.
        
    """
def deformable_convolution(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Create node performing deformable convolution.
    
        :param data: The node providing data batch tensor.
        :param filter: The node providing filters tensor.
        :param strides: The distance (in pixels) to slide the filter on the feature map over the axes.
        :param pads_begin: The number of pixels to add to the beginning along each axis.
        :param pads_end: The number of pixels to add to the end along each axis.
        :param dilations: The distance in width and height between elements (weights) in the filter.
        :param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
        :param group: The number of groups which both output and input should be split into.
        :param deformable_group: The number of groups which deformable values and output should be split
                                 into along the channel axis.
        :param name: The optional new name for output node.
        :return: New node performing deformable convolution operation.
        
    """
def deformable_psroi_pooling(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node performing DeformablePSROIPooling operation.
    
        DeformablePSROIPooling computes position-sensitive pooling
        on regions of interest specified by input.
    
        :param feature_maps: 4D tensor with feature maps.
        :param coords: 2D tensor describing box consisting of tuples: [batch_id, x_1, y_1, x_2, y_2].
        :param output_dim: A pooled output channel number.
        :param spatial_scale: A multiplicative spatial scale factor to translate ROI.
        :param group_size: The number of groups to encode position-sensitive score.
        :param mode: Specifies mode for pooling. Range of values: ['bilinear_deformable'].
        :param spatial_bins_x: Specifies numbers of bins to divide the input feature maps over width.
        :param spatial_bins_y: Specifies numbers of bins to divide the input feature maps over height.
        :param trans_std: The value that all transformation (offset) values are multiplied with.
        :param part_size: The number of parts the output tensor spatial dimensions are divided into.
        :param offsets: Optional node. 4D input blob with transformation values (offsets).
        :param name: The optional new name for output node.
        :return: New node performing DeformablePSROIPooling operation.
        
    """
def depth_to_space(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Rearranges input tensor from depth into blocks of spatial data.
    
        Values from the height and width dimensions are moved to the depth dimension.
    
        Input tensor has shape [N,C,H,W], where N is the batch axis, C is the channel or depth,
        H is the height and W is the width.
    
        Output node produces a tensor with shape:
    
        [N, C * `block_size` * `block_size`, H / `block_size`, W / `block_size`]
    
        :param node: The node with input tensor data.
        :param mode: Specifies how the input depth dimension is split to block coordinates
    
                     blocks_first: The input is divided to [block_size, ..., block_size, new_depth]
                     depth_first: The input is divided to [new_depth, block_size, ..., block_size]
    
        :param block_size: The size of the spatial block of values describing
                           how the tensor's data is to be rearranged.
        :param name: Optional output node name.
        :return: The new node performing an DepthToSpace operation on its input tensor.
        
    """
def detection_output(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Generate the detection output using information on location and confidence predictions.
    
        :param  box_logits:         The 2D input tensor with box logits.
        :param  class_preds:        The 2D input tensor with class predictions.
        :param  proposals:          The 3D input tensor with proposals.
        :param  attrs:              The dictionary containing key, value pairs for attributes.
        :param  aux_class_preds:    The 2D input tensor with additional class predictions information.
        :param  aux_box_preds:      The 2D input tensor with additional box predictions information.
        :param  name:               Optional name for the output node.
        :return: Node representing DetectionOutput operation.
    
         Available attributes are:
    
        * num_classes       The number of classes to be predicted.
                            Range of values: positive integer number
                            Default value: None
                            Required: yes
    
        * background_label_id   The background label id.
                                Range of values: integer value
                                Default value: 0
                                Required: no
    
        * top_k                 Maximum number of results to be kept per batch after NMS step.
                                Range of values: integer value
                                Default value: -1
                                Required: no
    
        * variance_encoded_in_target    The flag that denotes if variance is encoded in target.
                                        Range of values: {False, True}
                                        Default value: False
                                        Required: no
    
        * keep_top_k            Maximum number of bounding boxes per batch to be kept after NMS step.
                                Range of values: integer values
                                Default value: None
                                Required: yes
    
        * code_type             The type of coding method for bounding boxes.
                                Range of values: {'caffe.PriorBoxParameter.CENTER_SIZE',
                                                 'caffe.PriorBoxParameter.CORNER'}
    
                                Default value: 'caffe.PriorBoxParameter.CORNER'
                                Required: no
    
        * share_location        The flag that denotes if bounding boxes are shared among different
                                classes.
                                Range of values: {True, False}
                                Default value: True
                                Required: no
    
        * nms_threshold         The threshold to be used in the NMS stage.
                                Range of values: floating point value
                                Default value: None
                                Required: yes
    
        * confidence_threshold  Specifies the minimum confidence threshold for detection boxes to be
                                considered.
                                Range of values: floating point value
                                Default value: 0
                                Required: no
    
        * clip_after_nms        The flag that denotes whether to perform clip bounding boxes after
                                non-maximum suppression or not.
                                Range of values: {True, False}
                                Default value: False
                                Required: no
    
        * clip_before_nms       The flag that denotes whether to perform clip bounding boxes before
                                non-maximum suppression or not.
                                Range of values: {True, False}
                                Default value: False
                                Required: no
    
        * decrease_label_id     The flag that denotes how to perform NMS.
                                Range of values: False - perform NMS like in Caffe*.
                                                 True  - perform NMS like in MxNet*.
    
                                Default value: False
                                Required: no
    
        * normalized            The flag that denotes whether input tensors with boxes are normalized.
                                Range of values: {True, False}
                                Default value: False
                                Required: no
    
        * input_height          The input image height.
                                Range of values: positive integer number
                                Default value: 1
                                Required: no
    
        * input_width           The input image width.
                                Range of values: positive integer number
                                Default value: 1
                                Required: no
    
        * objectness_score      The threshold to sort out confidence predictions.
                                Range of values: non-negative float number
                                Default value: 0
                                Required: no
    
        Example of attribute dictionary:
        .. code-block:: python
    
            # just required ones
            attrs = {
                'num_classes': 85,
                'keep_top_k': [1, 2, 3],
                'nms_threshold': 0.645,
    
            }
    
            attrs = {
                'num_classes': 85,
                'keep_top_k': [1, 2, 3],
                'nms_threshold': 0.645,
                'normalized': True,
                'clip_before_nms': True,
                'input_height': [32],
                'input_width': [32],
    
            }
    
        Optional attributes which are absent from dictionary will be set with corresponding default.
        
    """
def divide(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies f(x) = A/B to the input nodes element-wise.
    
        :param left_node: The node providing dividend data.
        :param right_node: The node providing divisor data.
        :param auto_broadcast: Specifies rules used for auto-broadcasting of input tensors.
        :param name: Optional name for output node.
        :return: The node performing element-wise division.
        
    """
def elu(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform Exponential Linear Unit operation element-wise on data from input node.
    
        Computes exponential linear: alpha * (exp(data) - 1) if < 0, data otherwise.
    
        For more information refer to:
        [Fast and Accurate Deep Network Learning by Exponential Linear Units](http://arxiv.org/abs/1511.07289)
    
        :param data: Input tensor. One of: input node, array or scalar.
        :param alpha: Scalar multiplier for negative values.
        :param name: Optional output node name.
        :return: The new node performing an ELU operation on its input data element-wise.
        
    """
def equal(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which checks if input nodes are equal element-wise.
    
        :param left_node: The first input node for equal operation.
        :param right_node: The second input node for equal operation.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :param name: The optional name for output new node.
        :return: The node performing element-wise equality check.
        
    """
def erf(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which calculates Gauss error function element-wise with given tensor.
    
        :param node: The node providing data for operation.
        :param name: The optional name for new output node.
        :return: The new node performing element-wise Erf operation.
        
    """
def exp(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies exponential function to the input node element-wise.
    
        :param node: The node providing data for operation.
        :param name: The optional name for new output node.
        :return: The new node performing natural exponential operation.
        
    """
def fake_quantize(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform an element-wise linear quantization on input data.
    
        :param data:           The node with data tensor.
        :param input_low:      The node with the minimum for input values.
        :param input_high:     The node with the maximum for input values.
        :param output_low:     The node with the minimum quantized value.
        :param output_high:    The node with the maximum quantized value.
        :param levels:         The number of quantization levels. Integer value.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :return: New node with quantized value.
    
        Input floating point values are quantized into a discrete set of floating point values.
    
        .. code-block:: python
    
            if x <= input_low:
                output = output_low
            if x > input_high:
                output = output_high
            else:
                output = fake_quantize(output)
    
        Fake quantize uses the following logic:
    
        \\f[ output =
                \\dfrac{round( \\dfrac{data - input\\_low}{(input\\_high - input\\_low)\\cdot (levels-1)})}
                {(levels-1)\\cdot (output\\_high - output\\_low)} + output\\_low \\f]
        
    """
def floor(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies floor to the input node element-wise.
    
        :param node: The input node providing data.
        :param name: The optional name for new output node.
        :return: The node performing element-wise floor operation.
        
    """
def floor_mod(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node performing element-wise FloorMod (division reminder) with two given tensors.
    
        :param left_node: The first input node for FloorMod operation.
        :param right_node: The second input node for FloorMod operation.
        :param auto_broadcast: Specifies rules used for auto-broadcasting of input tensors.
        :param name: Optional name for output node.
        :return: The node performing element-wise FloorMod operation.
        
    """
def gather(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return Gather node which takes slices from axis of data according to indices.
    
        :param data: The tensor from which slices are gathered.
        :param indices: Tensor with indexes to gather.
        :param axis: The dimension index to gather data from.
        :param name: Optional name for output node.
        :return: The new node performing a Gather operation on the data input tensor.
        
    """
def gather_tree(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform GatherTree operation.
    
        :param step_ids: The tensor with indices from per each step.
        :param parent_idx: The tensor with with parent beam indices.
        :param max_seq_len: The tensor with maximum lengths for each sequence in the batch.
        :param end_token: The scalar tensor with value of the end marker in a sequence.
        :param name: Optional name for output node.
        :return: The new node performing a GatherTree operation.
    
        The GatherTree node generates the complete beams from the indices per each step
        and the parent beam indices.
        GatherTree uses the following logic:
    
        .. code-block:: python
    
            for batch in range(BATCH_SIZE):
                for beam in range(BEAM_WIDTH):
                    max_sequence_in_beam = min(MAX_TIME, max_seq_len[batch])
    
                    parent = parent_idx[max_sequence_in_beam - 1, batch, beam]
    
                    for level in reversed(range(max_sequence_in_beam - 1)):
                        final_idx[level, batch, beam] = step_idx[level, batch, parent]
    
                        parent = parent_idx[level, batch, parent]
        
    """
def greater(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which checks if left input node is greater than the right node element-wise.
    
        :param left_node: The first input node providing data.
        :param right_node: The second input node providing data.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :param name: The optional new name for output node.
        :return: The node performing element-wise check whether left_node is greater than right_node.
        
    """
def greater_equal(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which checks if left node is greater or equal to the right node element-wise.
    
        :param left_node: The first input node providing data.
        :param right_node: The second input node providing data.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :param name: The optional new name for output node.
    
        :return: The node performing element-wise check whether left_node is greater than or equal right_node.
        
    """
def grn(data: openvino._pyopenvino.Node, bias: float, name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Perform Global Response Normalization with L2 norm (across channels only).
    
        Computes GRN operation on channels for input tensor:
    
        \\f[ output_i = \\dfrac{input_i}{\\sqrt{\\sum_{i}^{C} input_i}} \\f]
    
        :param data: The node with data tensor.
        :param bias: The bias added to the variance. Scalar value.
        :param name: Optional output node name.
        :return: The new node performing a GRN operation on tensor's channels.
        
    """
def group_convolution(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform Group Convolution operation on data from input node.
    
        :param data:        The node producing input data.
        :param filters:     The node producing filters data.
        :param strides:     The distance (in pixels) to slide the filter on the feature map
                            over the axes.
        :param pads_begin:  The number of pixels to add at the beginning along each axis.
        :param pads_end:    The number of pixels to add at the end along each axis.
        :param dilations:   The distance in width and height between elements (weights) in the filter.
        :param auto_pad:    Describes how to perform padding. Possible values:
                            EXPLICIT:   Pad dimensions are explicity specified
                            SAME_LOWER: Pad dimensions computed to match input shape
                            Ceil(num_dims/2) at the beginning and
                            Floor(num_dims/2) at the end
    
                            SAME_UPPER: Pad dimensions computed to match input shape
                                        Floor(num_dims/2) at the beginning and
                                        Ceil(num_dims/2) at the end
    
                            VALID:      No padding
        :param name: Optional output node name.
        :return: The new node performing a Group Convolution operation on tensor from input node.
        
    """
def group_convolution_backprop_data(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform Group Convolution operation on data from input node.
    
        :param data:            The node producing input data.
        :param filters:         The node producing filter data.
        :param strides:         The distance (in pixels) to slide the filter on the feature map
                                over the axes.
        :param output_shape:    The node that specifies spatial shape of the output.
        :param pads_begin:      The number of pixels to add at the beginning along each axis.
        :param pads_end:        The number of pixels to add at the end along each axis.
        :param dilations:       The distance in width and height between elements (weights)
                                in the filter.
        :param auto_pad:        Describes how to perform padding. Possible values:
                                EXPLICIT:   Pad dimensions are explicity specified
                                SAME_LOWER: Pad dimensions computed to match input shape
                                Ceil(num_dims/2) at the beginning and
                                Floor(num_dims/2) at the end
    
                                SAME_UPPER: Pad dimensions computed to match input shape
                                            Floor(num_dims/2) at the beginning and
                                            Ceil(num_dims/2) at the end
    
                                VALID:      No padding
    
        :param output_padding:  The additional amount of paddings added per each spatial axis
                                in the output tensor.
        :param name: Optional output node name.
        :return: The new node performing a Group Convolution operation on tensor from input node.
        
    """
def hard_sigmoid(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform Hard Sigmoid operation element-wise on data from input node.
    
        :param data: The node with data tensor.
        :param alpha: A node producing the alpha parameter.
        :param beta: A node producing the beta parameter
        :param name: Optional output node name.
        :return: The new node performing a Hard Sigmoid element-wise on input tensor.
    
        Hard Sigmoid uses the following logic:
    
        .. code-block:: python
    
            y = max(0, min(1, alpha * data + beta))
        
    """
def interpolate(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform interpolation of independent slices in input tensor.
    
        :param  image:         The node providing input tensor with data for interpolation.
        :param  output_shape:  1D tensor describing output shape for spatial axes.
        :param  attrs:         The dictionary containing key, value pairs for attributes.
        :param  name:          Optional name for the output node.
        :return: Node representing interpolation operation.
    
        Available attributes are:
    
        * axes              Specify spatial dimension indices where interpolation is applied.
                            Type: List of non-negative integer numbers.
                            Required: yes.
    
        * mode              Specifies type of interpolation.
                            Range of values: one of {nearest, linear, cubic, area}
                            Type: string
                            Required: yes
    
        * align_corners     A flag that specifies whether to align corners or not. True means the
                            alignment is applied, False means the alignment isn't applied.
                            Range of values: True or False. Default: True.
                            Required: no
    
        * antialias         A flag that specifies whether to perform anti-aliasing.
                            Range of values: False - do not perform anti-aliasing
                                             True - perform anti-aliasing
    
                            Default value: False
                            Required: no
    
        * pads_begin        Specify the number of pixels to add to the beginning of the image being
                            interpolated. A scalar that specifies padding for each spatial dimension.
                            Range of values: list of non-negative integer numbers. Default value: 0
                            Required: no
    
        * pads_end          Specify the number of pixels to add to the beginning of the image being
                            interpolated. A scalar that specifies padding for each spatial dimension.
                            Range of values: list of non-negative integer numbers. Default value: 0
                            Required: no
    
        Example of attribute dictionary:
    
        .. code-block:: python
    
            # just required ones
            attrs = {
                'axes': [2, 3],
                'mode': 'cubic',
            }
    
            attrs = {
                'axes': [2, 3],
                'mode': 'cubic',
                'antialias': True,
                'pads_begin': [2, 2, 2],
            }
    
        Optional attributes which are absent from dictionary will be set with corresponding default.
        
    """
def less(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which checks if left input node is less than the right node element-wise.
    
        :param left_node: The first input node providing data.
        :param right_node: The second input node providing data.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :param name: The optional new name for output node.
        :return: The node performing element-wise check whether left_node is less than the right_node.
        
    """
def less_equal(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which checks if left input node is less or equal the right node element-wise.
    
        :param left_node: The first input node providing data.
        :param right_node: The second input node providing data.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :param name: The optional new name for output node.
        :return: The node performing element-wise check whether left_node is less than or equal the
                 right_node.
        
    """
def log(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies natural logarithm to the input node element-wise.
    
        :param node: The input node providing data for operation.
        :param name: The optional new name for output node.
        :return: The new node performing log operation element-wise.
        
    """
def logical_and(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which perform logical and operation on input nodes element-wise.
    
        :param left_node: The first input node providing data.
        :param right_node: The second input node providing data.
        :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                               to output shape axes. Range of values: numpy, explicit.
        :param name: The optional new name for output node.
        :return: The node performing logical and operation on input nodes corresponding elements.
        
    """
def logical_not(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies element-wise logical negation to the input node.
    
        :param node: The input node providing data.
        :param name: The optional new name for output node.
        :return: The node performing element-wise logical NOT operation with given tensor.
        
    """
def logical_or(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs logical OR operation on input nodes element-wise.
    
        :param left_node: The first input node providing data.
        :param right_node: The second input node providing data.
        :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                               to output shape axes. Range of values: numpy, explicit.
        :param name: The optional new name for output node.
        :return: The node performing logical or operation on input nodes corresponding elements.
        
    """
def logical_xor(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs logical XOR operation on input nodes element-wise.
    
        :param left_node: The first input node providing data.
        :param right_node: The second input node providing data.
        :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                               to output shape axes. Range of values: numpy, explicit.
        :param name: The optional new name for output node.
        :return: The node performing logical or operation on input nodes corresponding elements.
        
    """
def lrn(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs element-wise Local Response Normalization (LRN) operation.
    
        :param data: Input data.
        :param alpha: A scale factor (usually positive).
        :param beta: An exponent.
        :param bias: An offset (usually positive) to avoid dividing by 0.
        :param size: Width of the 1-D normalization window.
        :param name: An optional name of the output node.
        :return: The new node which performs LRN.
        
    """
def lstm_cell(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs LSTMCell operation.
    
        :param X: The input tensor with shape: [batch_size, input_size].
        :param initial_hidden_state: The hidden state tensor with shape: [batch_size, hidden_size].
        :param initial_cell_state: The cell state tensor with shape: [batch_size, hidden_size].
        :param W: The weight tensor with shape: [4*hidden_size, input_size].
        :param R: The recurrence weight tensor with shape: [4*hidden_size, hidden_size].
        :param B: The bias tensor for gates with shape: [4*hidden_size].
        :param hidden_size: Specifies hidden state size.
        :param activations: The list of three activation functions for gates.
        :param activations_alpha: The list of alpha parameters for activation functions.
        :param activations_beta: The list of beta parameters for activation functions.
        :param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
        :param name: An optional name of the output node.
    
        :return: The new node represents LSTMCell. Node outputs count: 2.
        
    """
def matmul(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return the Matrix Multiplication operation.
    
        :param data_a: left-hand side matrix
        :param data_b: right-hand side matrix
        :param transpose_a: should the first matrix be transposed before operation
        :param transpose_b: should the second matrix be transposed
        :return: MatMul operation node
        
    """
def max_pool(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform max pooling operation with given parameters on provided data.
    
        :param  data:           The node providing input data.
        :param  strides:        The distance (in pixels) to slide the filter on the feature map
                                over the axes.
        :param  pads_begin:     The number of pixels to add at the beginning along each axis.
        :param  pads_end:       The number of pixels to add at the end along each axis.
        :param  kernel_shape:   The pooling operation kernel shape.
        :param  rounding_type:  Determines used rounding schema when computing output shape. Acceptable
                                values are: ['floor', 'ceil']
        :param  auto_pad:       Determines how the padding is calculated. Acceptable values:
                                [None, 'same_upper', 'same_lower', 'valid']
        :param  name:           The optional name for the created output node.
    
        :return:   The new node performing max pooling operation.
        
    """
def maximum(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies the maximum operation to input nodes elementwise.
    
        :param left_node: The first input node for maximum operation.
        :param right_node: The second input node for maximum operation.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors. Defaults to "NUMPY".
        :param name: The optional name for output new node.
        :return: The node performing element-wise maximum operation.
        
    """
def minimum(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies the minimum operation to input nodes elementwise.
    
        :param left_node: The first input node for minimum operation.
        :param right_node: The second input node for minimum operation.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors. Defaults to "NUMPY".
        :param name: The optional name for output new node.
        :return: The node performing element-wise minimum operation.
        
    """
def mod(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node performing element-wise division reminder with two given tensors.
    
        :param left_node: The first input node for mod operation.
        :param right_node: The second input node for mod operation.
        :param auto_broadcast: Specifies rules used for auto-broadcasting of input tensors.
        :param name: Optional name for output node.
        :return: The node performing element-wise Mod operation.
        
    """
def multiply(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies f(A,B) = A*B to the input nodes elementwise.
    
        :param left_node: The first input node for multiply operation.
        :param right_node: The second input node for multiply operation.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors. Defaults to "NUMPY".
        :param name: The optional name for output new node.
        :return: The node performing element-wise multiplication.
        
    """
def negative(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies f(x) = -x to the input node elementwise.
    
        :param node: Input node for negative operation.
        :param name: The optional name for output new node.
        :return: The node performing element-wise multiplicaion by -1.
        
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
        :param box_encoding: Format of boxes data encoding. Range of values: corner or cente.
        :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                       boxes across batches or not.
        :return: The new node which performs NonMaxSuppression
        
    """
def normalize_l2(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Construct an NormalizeL2 operation.
    
        :param data: Node producing the input tensor
        :param axes: Node indicating axes along which L2 reduction is calculated
        :param eps: The epsilon added to L2 norm
        :param eps_mode: how eps is combined with L2 value (`add` or `max`)
        :return: New node which performs the L2 normalization.
        
    """
def not_equal(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which checks if input nodes are unequal element-wise.
    
        :param left_node: The first input node for not-equal operation.
        :param right_node: The second input node for not-equal operation.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :param name: The optional name for output new node.
        :return: The node performing element-wise inequality check.
        
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
    
        :param name: The optional name for new output node.
        :return: New node performing one-hot operation.
        
    """
def pad(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a generic padding operation.
    
        :param arg: The node producing input tensor to be padded.
        :param pads_begin: number of padding elements to be added before position 0
                           on each axis of arg.
        :param pads_end: number of padding elements to be added after the last element.
        :param pad_mode: "constant", "edge", "reflect" or "symmetric"
        :param arg_pad_value: value used for padding if pad_mode is "constant"
        :return: Pad operation node.
        
    """
def parameter(*args, **kwargs) -> openvino._pyopenvino.op.Parameter:
    """
    Return an openvino Parameter object.
    
        :param shape: The shape of the output tensor.
        :param dtype: The type of elements of the output tensor. Defaults to np.float32.
        :param name: The optional name for output new node.
        :return: The node that specifies input to the model.
        
    """
def power(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which perform element-wise exponentiation operation.
    
        :param left_node: The node providing the base of operation.
        :param right_node: The node providing the exponent of operation.
        :param name: The optional name for the new output node.
        :param auto_broadcast: The type of broadcasting specifies rules used for
                               auto-broadcasting of input tensors.
        :return: The new node performing element-wise exponentiation operation on input nodes.
        
    """
def prelu(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform Parametrized Relu operation element-wise on data from input node.
    
        :param data: The node with data tensor.
        :param slope: The node with the multipliers for negative values.
        :param name: Optional output node name.
        :return: The new node performing a PRelu operation on tensor's channels.
    
        PRelu uses the following logic:
    
        .. code-block:: python
    
            if data < 0:
                data = data * slope
            elif data >= 0:
                data = data
        
    """
def prior_box(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Generate prior boxes of specified sizes and aspect ratios across all dimensions.
    
        :param  layer_shape:  Shape of layer for which prior boxes are computed.
        :param  image_shape:  Shape of image to which prior boxes are scaled.
        :param  attrs:        The dictionary containing key, value pairs for attributes.
        :param  name:         Optional name for the output node.
        :return: Node representing prior box operation.
    
        Available attributes are:
    
        * min_size          The minimum box size (in pixels).
                            Range of values: positive floating point numbers
                            Default value: []
                            Required: no
    
        * max_size          The maximum box size (in pixels).
                            Range of values: positive floating point numbers
                            Default value: []
                            Required: no
    
        * aspect_ratio      Aspect ratios of prior boxes.
                            Range of values: set of positive floating point numbers
                            Default value: []
                            Required: no
    
        * flip              The flag that denotes that each aspect_ratio is duplicated and flipped.
                            Range of values: {True, False}
                            Default value: False
                            Required: no
    
        * clip              The flag that denotes if each value in the output tensor should be clipped
                            to [0,1] interval.
                            Range of values: {True, False}
                            Default value: False
                            Required: no
    
        * step              The distance between box centers.
                            Range of values: floating point non-negative number
                            Default value: 0
                            Required: no
    
        * offset            This is a shift of box respectively to top left corner.
                            Range of values: floating point non-negative number
                            Default value: None
                            Required: yes
    
        * variance          The variance denotes a variance of adjusting bounding boxes. The attribute
                            could contain 0, 1 or 4 elements.
                            Range of values: floating point positive numbers
                            Default value: []
                            Required: no
    
        * scale_all_sizes   The flag that denotes type of inference.
                            Range of values: False - max_size is ignored
                                             True  - max_size is used
    
                            Default value: True
                            Required: no
    
        * fixed_ratio       This is an aspect ratio of a box.
                            Range of values: a list of positive floating-point numbers
                            Default value: None
                            Required: no
    
        * fixed_size        This is an initial box size (in pixels).
                            Range of values: a list of positive floating-point numbers
                            Default value: None
                            Required: no
    
        * density           This is the square root of the number of boxes of each type.
                            Range of values: a list of positive floating-point numbers
                            Default value: None
                            Required: no
    
        Example of attribute dictionary:
    
        .. code-block:: python
    
            # just required ones
            attrs = {
                'offset': 85,
            }
    
            attrs = {
                'offset': 85,
                'flip': True,
                'clip': True,
                'fixed_size': [32, 64, 128]
            }
    
        Optional attributes which are absent from dictionary will be set with corresponding default.
        
    """
def prior_box_clustered(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Generate prior boxes of specified sizes normalized to the input image size.
    
        :param  output_size:    1D tensor with two integer elements [height, width]. Specifies the
                                spatial size of generated grid with boxes.
        :param  image_size:     1D tensor with two integer elements [image_height, image_width] that
                                specifies shape of the image for which boxes are generated.
        :param  attrs:          The dictionary containing key, value pairs for attributes.
        :param  name:           Optional name for the output node.
        :return: Node representing PriorBoxClustered operation.
    
         Available attributes are:
    
        * widths        Specifies desired boxes widths in pixels.
                        Range of values: floating point positive numbers.
                        Default value: 1.0
                        Required: no
    
        * heights       Specifies desired boxes heights in pixels.
                        Range of values: floating point positive numbers.
                        Default value: 1.0
                        Required: no
    
        * clip          The flag that denotes if each value in the output tensor should be clipped
                        within [0,1].
                        Range of values: {True, False}
                        Default value: True
                        Required: no
    
        * step_widths   The distance between box centers.
                        Range of values: floating point positive number
                        Default value: 0.0
                        Required: no
    
        * step_heights  The distance between box centers.
                        Range of values: floating point positive number
                        Default value: 0.0
                        Required: no
    
        * offset        The shift of box respectively to the top left corner.
                        Range of values: floating point positive number
                        Default value: None
                        Required: yes
    
        * variance      Denotes a variance of adjusting bounding boxes.
                        Range of values: floating point positive numbers
                        Default value: []
                        Required: no
    
        Example of attribute dictionary:
    
        .. code-block:: python
    
            # just required ones
            attrs = {
                'offset': 85,
            }
    
            attrs = {
                'offset': 85,
                'clip': False,
                'step_widths': [1.5, 2.0, 2.5]
            }
    
        Optional attributes which are absent from dictionary will be set with corresponding default.
        
    """
def proposal(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Filter bounding boxes and outputs only those with the highest prediction confidence.
    
        :param  class_probs:        4D input floating point tensor with class prediction scores.
        :param  bbox_deltas:         4D input floating point tensor with box logits.
        :param  image_shape:        The 1D input tensor with 3 or 4 elements describing image shape.
        :param  attrs:              The dictionary containing key, value pairs for attributes.
        :param  name:               Optional name for the output node.
    
        :return: Node representing Proposal operation.
    
        * base_size     The size of the anchor to which scale and ratio attributes are applied.
                        Range of values: a positive unsigned integer number
                        Default value: None
                        Required: yes
    
        * pre_nms_topn  The number of bounding boxes before the NMS operation.
                        Range of values: a positive unsigned integer number
                        Default value: None
                        Required: yes
    
        * post_nms_topn The number of bounding boxes after the NMS operation.
                        Range of values: a positive unsigned integer number
                        Default value: None
                        Required: yes
    
        * nms_thresh    The minimum value of the proposal to be taken into consideration.
                        Range of values: a positive floating-point number
                        Default value: None
                        Required: yes
    
        * feat_stride   The step size to slide over boxes (in pixels).
                        Range of values: a positive unsigned integer
                        Default value: None
                        Required: yes
    
        * min_size      The minimum size of box to be taken into consideration.
                        Range of values: a positive unsigned integer number
                        Default value: None
                        Required: yes
    
        * ratio         The ratios for anchor generation.
                        Range of values: a list of floating-point numbers
                        Default value: None
                        Required: yes
    
        * scale         The scales for anchor generation.
                        Range of values: a list of floating-point numbers
                        Default value: None
                        Required: yes
    
        * clip_before_nms   The flag that specifies whether to perform clip bounding boxes before
                            non-maximum suppression or not.
                            Range of values: True or False
                            Default value: True
                            Required: no
    
        * clip_after_nms    The flag that specifies whether to perform clip bounding boxes after
                            non-maximum suppression or not.
                            Range of values: True or False
                            Default value: False
                            Required: no
    
        * normalize     The flag that specifies whether to perform normalization of output boxes to
                        [0,1] interval or not.
                        Range of values: True or False
                        Default value: False
                        Required: no
    
        * box_size_scale    Specifies the scale factor applied to logits of box sizes before decoding.
                            Range of values: a positive floating-point number
                            Default value: 1.0
                            Required: no
    
        * box_coordinate_scale  Specifies the scale factor applied to logits of box coordinates
                                before decoding.
                                Range of values: a positive floating-point number
                                Default value: 1.0
                                Required: no
    
        * framework     Specifies how the box coordinates are calculated.
                        Range of values: "" (empty string) - calculate box coordinates like in Caffe*
                                         tensorflow - calculate box coordinates like in the TensorFlow*
                                                      Object Detection API models
    
                        Default value: "" (empty string)
                        Required: no
    
        Example of attribute dictionary:
    
        .. code-block:: python
    
            # just required ones
            attrs = {
                'base_size': 85,
                'pre_nms_topn': 10,
                'post_nms_topn': 20,
                'nms_thresh': 0.34,
                'feat_stride': 16,
                'min_size': 32,
                'ratio': [0.1, 1.5, 2.0, 2.5],
                'scale': [2, 3, 3, 4],
            }
    
        Optional attributes which are absent from dictionary will be set with corresponding default.
    
        
    """
def psroi_pooling(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a PSROIPooling operation.
    
        :param input: Input feature map `{N, C, ...}`.
        :param coords: Coordinates of bounding boxes.
        :param output_dim: Output channel number.
        :param group_size: Number of groups to encode position-sensitive scores.
        :param spatial_scale: Ratio of input feature map over input image size.
        :param spatial_bins_x: Numbers of bins to divide the input feature maps over.
        :param spatial_bins_y: Numbers of bins to divide the input feature maps over.
        :param mode: Mode of pooling - "avg" or "bilinear".
        :return: PSROIPooling node
        
    """
def range(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the Range operation.
    
        :param start:  The start value of the generated range.
        :param stop:   The stop value of the generated range.
        :param step:   The step value for the generated range.
        :param name:   Optional name for output node.
        :return: Range node
        
    """
def reduce_logical_and(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Logical AND reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to reduce.
        :param reduction_axes: The axes to eliminate through AND operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction.
        :param name:           Optional name for output node.
        :return: The new node performing reduction operation.
        
    """
def reduce_logical_or(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Logical OR reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to reduce.
        :param reduction_axes: The axes to eliminate through OR operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction.
        :param name:           Optional name for output node.
        :return: The new node performing reduction operation.
        
    """
def reduce_max(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Max-reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to max-reduce.
        :param reduction_axes: The axes to eliminate through max operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction.
        :param name: Optional name for output node.
        
    """
def reduce_mean(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Mean-reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to mean-reduce.
        :param reduction_axes: The axes to eliminate through mean operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction.
        :param name:           Optional name for output node.
        :return: The new node performing mean-reduction operation.
        
    """
def reduce_min(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Min-reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to min-reduce.
        :param reduction_axes: The axes to eliminate through min operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction
        :param name:           Optional name for output node.
        
    """
def reduce_prod(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Product-reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to product-reduce.
        :param reduction_axes: The axes to eliminate through product operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction
        :param name:           Optional name for output node.
        :return: The new node performing product-reduction operation.
        
    """
def reduce_sum(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform element-wise sums of the input tensor, eliminating the specified reduction axes.
    
        :param node:           The node providing data for operation.
        :param reduction_axes: The axes to eliminate through summation.
        :param keep_dims:      If set to True it holds axes that are used for reduction
        :param name:           The optional new name for output node.
        :return: The new node performing summation along `reduction_axes` element-wise.
        
    """
def region_yolo(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the RegionYolo operation.
    
        :param input:       Input data
        :param coords:      Number of coordinates for each region
        :param classes:     Number of classes for each region
        :param num:         Number of regions
        :param do_softmax:  Compute softmax
        :param mask:        Mask
        :param axis:        Axis to begin softmax on
        :param end_axis:    Axis to end softmax on
        :param anchors:     A flattened list of pairs `[width, height]` that describes prior box sizes
        :param name:        Optional name for output node.
        :return: RegionYolo node
        
    """
def relu(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform rectified linear unit operation on input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: The optional output node name.
        :return: The new node performing relu operation on its input element-wise.
        
    """
def reshape(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return reshaped node according to provided parameters.
    
        :param node: The tensor we want to reshape.
        :param output_shape: The node with a new shape for input tensor.
        :param special_zero: The boolean variable that controls how zero values in shape are
                             interpreted. If special_zero is false, then 0 is interpreted as-is
                             which means that output shape will contain a zero dimension at the
                             specified location. Input and output tensors are empty in this case.
                             If special_zero is true, then all zeros in shape implies the copying
                             of corresponding dimensions from data.shape into the output shape.
                             Range of values: False or True
        :return: The node reshaping an input tensor.
        
    """
def result(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which represents an output of a graph (Model).
    
        :param data: The tensor containing the input data
        :return: Result node
        
    """
def reverse_sequence(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a ReverseSequence operation.
    
        :param input: tensor with input data to reverse
        :param seq_lengths: 1D tensor of integers with sequence lengths in the input tensor.
        :param batch_axis: index of the batch dimension.
        :param seq_axis: index of the sequence dimension.
        :return: ReverseSequence node
        
    """
def select(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform an element-wise selection operation on input tensors.
    
        :param cond: Tensor with selection mask of type `boolean`.
        :param then_node: Tensor providing data to be selected if respective `cond`
                            item value is `True`.
        :param else_node: Tensor providing data to be selected if respective `cond`
                            item value is `False`.
        :param auto_broadcast: Mode specifies rules used for auto-broadcasting of input tensors.
        :param name: The optional new name for output node.
        :return: The new node with values selected according to provided arguments.
        
    """
def selu(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform a Scaled Exponential Linear Unit (SELU) operation on input node element-wise.
    
        :param data: input node, array or scalar.
        :param alpha: Alpha coefficient of SELU operation
        :param lambda_value: Lambda coefficient of SELU operation
        :param name: The optional output node name.
        :return: The new node performing relu operation on its input element-wise.
        
    """
def shape_of(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces a tensor containing the shape of its input data.
    
        :param data: The tensor containing the input data.
        :return: ShapeOf node
        
    """
def sigmoid(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which applies the sigmoid function element-wise.
    
        :param data: The tensor containing the input data
        :return: Sigmoid node
        
    """
def sign(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform element-wise sign operation.
    
        :param node: One of: input node, array or scalar.
        :param name: The optional new name for output node.
        :return: The node with mapped elements of the input tensor to -1 (if it is negative),
                 0 (if it is zero), or 1 (if it is positive).
        
    """
def sin(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply sine function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with sin operation applied on it.
        
    """
def sinh(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply hyperbolic sine function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with sin operation applied on it.
        
    """
def softmax(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply softmax operation on each element of input tensor.
    
        :param data: The tensor providing input data.
        :param axis: An axis along which Softmax should be calculated
        :return: The new node with softmax operation applied on each element.
        
    """
def space_to_depth(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform SpaceToDepth operation on the input tensor.
    
        SpaceToDepth rearranges blocks of spatial data into depth.
        The operator :return: a copy of the input tensor where values from the height
        and width dimensions are moved to the depth dimension.
    
        :param data: The node with data tensor.
        :param mode: Specifies how the output depth dimension is gathered from block coordinates.
    
                     blocks_first: The output depth is gathered from [block_size, ..., block_size, C]
                     depth_first: The output depth is gathered from [C, block_size, ..., block_size]
    
        :param block_size: The size of the block of values to be moved. Scalar value.
        :param name: Optional output node name.
        :return: The new node performing a SpaceToDepth operation on input tensor.
        
    """
def split(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which splits the input tensor into same-length slices.
    
        :param data: The input tensor to be split
        :param axis: Axis along which the input data will be split
        :param num_splits: Number of the output tensors that should be produced
        :return: Split node
        
    """
def sqrt(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies square root to the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: The new node with sqrt operation applied element-wise.
        
    """
def squared_difference(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform an element-wise squared difference between two tensors.
    
        \\f[ y[i] = (x_1[i] - x_2[i])^2 \\f]
    
        :param x1: The node with first input tensor.
        :param x2: The node with second input tensor.
        :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                               to output shape axes. Range of values: numpy, explicit.
        :param name: Optional new name for output node.
        :return: The new node performing a squared difference between two tensors.
        
    """
def squeeze(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform squeeze operation on input tensor.
    
        :param data: The node with data tensor.
        :param axes: List of non-negative integers, indicate the dimensions to squeeze.
                      One of: input node or array.
        :param name: Optional new name for output node.
        :return: The new node performing a squeeze operation on input tensor.
    
        Remove single-dimensional entries from the shape of a tensor.
        Takes a parameter `axes` with a list of axes to squeeze.
        If `axes` is not provided, all the single dimensions will be removed from the shape.
        If an `axis` is selected with shape entry not equal to one, an error is raised.
    
    
        For example:
    
           Inputs: tensor with shape [1, 2, 1, 3, 1, 1], axes=[2, 4]
    
           Result: tensor with shape [1, 2, 3, 1]
        
    """
def strided_slice(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which dynamically repeats(replicates) the input data tensor.
    
        :param      data:              The tensor to be sliced
        :param      begin:             1D tensor with begin indexes for input blob slicing
        :param      end:               1D tensor with end indexes for input blob slicing
        :param      strides:           The slicing strides
        :param      begin_mask:        A mask applied to the 'begin' input indicating which elements
                                       shoud be ignored
        :param      end_mask:          A mask applied to the 'end' input indicating which elements
                                       shoud be ignored
        :param      new_axis_mask:     A mask indicating dimensions where '1' should be inserted
        :param      shrink_axis_mask:  A mask indicating which dimensions should be deleted
        :param      ellipsis_mask:     Indicates positions where missing dimensions should be inserted
        :return:   StridedSlice node
        
    """
def subtract(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies f(x) = A-B to the input nodes element-wise.
    
        :param left_node: The node providing data for left hand side of operator.
        :param right_node: The node providing data for right hand side of operator.
        :param auto_broadcast: The type of broadcasting that specifies mapping of input tensor axes
                               to output shape axes. Range of values: numpy, explicit.
        :param name: The optional name for output node.
        :return: The new output node performing subtraction operation on both tensors element-wise.
        
    """
def tan(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply tangent function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with tan operation applied on it.
        
    """
def tanh(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which applies hyperbolic tangent to the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with tanh operation applied on it.
        
    """
def tile(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which dynamically repeats(replicates) the input data tensor.
    
        :param data: The input tensor to be tiled
        :param repeats: Per-dimension replication factors
        :return: Tile node
        
    """
def topk(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs TopK.
    
        :param data: Input data.
        :param k: K.
        :param axis: TopK Axis.
        :param mode: Compute TopK largest ('max') or smallest ('min')
        :param sort: Order of output elements (sort by: 'none', 'index' or 'value')
        :return: The new node which performs TopK (both indices and values)
        
    """
def transpose(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which transposes the data in the input tensor.
    
        :param data: The input tensor to be transposed
        :param input_order: Permutation of axes to be applied to the input tensor
        :return: Transpose node
        
    """
def unsqueeze(data: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], axes: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Perform unsqueeze operation on input tensor.
    
        Insert single-dimensional entries to the shape of a tensor. Takes one required argument axes,
        a list of dimensions that will be inserted.
        Dimension indices in axes are as seen in the output tensor.
    
        For example: Inputs: tensor with shape [3, 4, 5], axes=[0, 4]
                     Result: tensor with shape [1, 3, 4, 5, 1]
    
        :param data: The node with data tensor.
        :param axes: List of non-negative integers, indicate the dimensions to be inserted.
                      One of: input node or array.
        :return: The new node performing an unsqueeze operation on input tensor.
        
    """
def variadic_split(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which splits the input tensor into variadic length slices.
    
        :param data: The input tensor to be split
        :param axis: Axis along which the input data will be split
        :param split_lengths: Sizes of the output tensors along the split axis
        :return: VariadicSplit node
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset1: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset1')
