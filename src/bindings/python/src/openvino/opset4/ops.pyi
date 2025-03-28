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
__all__ = ['Constant', 'Node', 'NodeFactory', 'NodeInput', 'NumericData', 'NumericType', 'Parameter', 'ScalarData', 'Shape', 'TensorShape', 'acosh', 'as_node', 'as_nodes', 'asinh', 'assert_list_of_ints', 'atanh', 'binary_op', 'check_valid_attributes', 'ctc_loss', 'get_dtype', 'get_element_type', 'get_element_type_str', 'hswish', 'is_non_negative_value', 'is_positive_value', 'lstm_cell', 'make_constant_node', 'mish', 'nameable_op', 'non_max_suppression', 'np', 'partial', 'proposal', 'range', 'reduce_l1', 'reduce_l2', 'scatter_nd_update', 'softplus', 'swish', 'unary_op']
def acosh(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply hyperbolic inverse cosine function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with arccosh operation applied on it.
        
    """
def asinh(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply hyperbolic inverse sinus function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with arcsinh operation applied on it.
        
    """
def atanh(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply hyperbolic inverse tangent function on the input node element-wise.
    
        :param node: One of: input node, array or scalar.
        :param name: Optional new name for output node.
        :return: New node with arctanh operation applied on it.
        
    """
def ctc_loss(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs CTCLoss.
    
        :param logits:                        3-D tensor of logits.
        :param logit_length:                  1-D tensor of lengths for each object from a batch.
        :param labels:                        2-D tensor of labels for which likelihood is estimated using logits.
        :param label_length:                  1-D tensor of length for each label sequence.
        :param blank_index:                   Scalar used to mark a blank index.
        :param preprocess_collapse_repeated:  Flag for preprocessing labels before loss calculation.
        :param ctc_merge_repeated:            Flag for merging repeated characters in a potential alignment.
        :param unique:                        Flag to find unique elements in a target.
        :return: The new node which performs CTCLoss
        
    """
def hswish(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs HSwish (hard version of Swish).
    
        :param data: Tensor with input data floating point type.
        :return: The new node which performs HSwish
        
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
def mish(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs Mish.
    
        :param data: Tensor with input data floating point type.
        :return: The new node which performs Mish
        
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
def proposal(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Filter bounding boxes and outputs only those with the highest prediction confidence.
    
        :param  class_probs:        4D input floating point tensor with class prediction scores.
        :param  bbox_deltas:        4D input floating point tensor with corrected predictions of bounding boxes
        :param  image_shape:        The 1D input tensor with 3 or 4 elements describing image shape.
        :param  attrs:              The dictionary containing key, value pairs for attributes.
        :param  name:               Optional name for the output node.
    
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
        :return: Node representing Proposal operation.
        
    """
def range(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the Range operation.
    
        :param start:       The start value of the generated range.
        :param stop:        The stop value of the generated range.
        :param step:        The step value for the generated range.
        :param output_type: The output tensor type.
        :param name:        Optional name for output node.
        :return: Range node
        
    """
def reduce_l1(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    L1-reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to mean-reduce.
        :param reduction_axes: The axes to eliminate through mean operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction
        :param name:           Optional name for output node.
        :return: The new node performing mean-reduction operation.
        
    """
def reduce_l2(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    L2-reduction operation on input tensor, eliminating the specified reduction axes.
    
        :param node:           The tensor we want to mean-reduce.
        :param reduction_axes: The axes to eliminate through mean operation.
        :param keep_dims:      If set to True it holds axes that are used for reduction
        :param name:           Optional name for output node.
        :return: The new node performing mean-reduction operation.
        
    """
def scatter_nd_update(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs ScatterNDUpdate.
    
        :param data: Node input representing the tensor to be updated.
        :param indices: Node input representing the indices at which updates will be applied.
        :param updates: Node input representing the updates to be applied.
        :param name: Optional name for the output node.
        :return: New node performing the ScatterNDUpdate.
        
    """
def softplus(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply SoftPlus operation on each element of input tensor.
    
        :param data: The tensor providing input data.
        :return: The new node with SoftPlus operation applied on each element.
        
    """
def swish(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performing Swish activation function Swish(x, beta=1.0) = x * sigmoid(x * beta)).
    
        :param data: Tensor with input data floating point type.
        :return: The new node which performs Swish
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset4: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset4')
