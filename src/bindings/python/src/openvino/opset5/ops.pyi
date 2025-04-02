# type: ignore
"""
Factory functions for all openvino ops.
"""
from functools import partial
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino._pyopenvino import Shape
from openvino._pyopenvino.op import Constant
from openvino._pyopenvino.op import loop
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
__all__ = ['Constant', 'Node', 'NodeFactory', 'NodeInput', 'NumericData', 'NumericType', 'Parameter', 'ScalarData', 'Shape', 'TensorShape', 'as_node', 'as_nodes', 'assert_list_of_ints', 'batch_norm_inference', 'binary_op', 'check_valid_attributes', 'gather_nd', 'get_dtype', 'get_element_type', 'get_element_type_str', 'gru_sequence', 'hsigmoid', 'is_non_negative_value', 'is_positive_value', 'log_softmax', 'loop', 'lstm_sequence', 'make_constant_node', 'nameable_op', 'non_max_suppression', 'np', 'partial', 'rnn_sequence', 'round', 'unary_op']
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
def gather_nd(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs GatherND.
    
        :param data:       N-D tensor with data for gathering
        :param indices:    K-D tensor of tuples with indices by which data is gathered
        :param batch_dims: Scalar value of batch dimensions
        :return: The new node which performs GatherND
        
    """
def gru_sequence(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs GRUSequence operation.
    
        :param inputs: The input tensor. Shape: [batch_size, seq_length, input_size].
        :param initial_hidden_state:    The hidden state tensor.
                                        Shape: [batch_size, num_directions, hidden_size].
        :param sequence_lengths:        Specifies real sequence lengths for each batch element.
                                        Shape: [batch_size]. Integer type.
        :param weights_w: Tensor with weights for matrix multiplication operation with input portion of data.
                  Shape: [num_directions, 3*hidden_size, input_size].
        :param weights_r: The tensor with weights for matrix multiplication operation with hidden state.
                  Shape: [num_directions, 3*hidden_size, hidden_size].
        :param biases: The sum of biases (weight and recurrence).
                  For linear_before_reset set True the shape is [num_directions, 4*hidden_size].
                  Otherwise the shape is [num_directions, 3*hidden_size].
        :param hidden_size: Specifies hidden state size.
        :param direction: Specifies if the RNN is forward, reverse, or bidirectional.
        :param activations: The list of three activation functions for gates.
        :param activations_alpha: The list of alpha parameters for activation functions.
        :param activations_beta: The list of beta parameters for activation functions.
        :param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
        :param linear_before_reset: Flag denotes if the layer behaves according to the modification
                                    of GRU described in the formula in the ONNX documentation.
        :param name: An optional name of the output node.
    
        :return: The new node represents GRUSequence. Node outputs count: 2.
        
    """
def hsigmoid(data: typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray], name: typing.Optional[str] = None) -> openvino._pyopenvino.Node:
    """
    Return a node which performs HSigmoid.
    
        :param data: Tensor with input data floating point type.
        :return: The new node which performs HSigmoid
        
    """
def log_softmax(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply LogSoftmax operation on each element of input tensor.
    
        :param data: The tensor providing input data.
        :param axis: An axis along which LogSoftmax should be calculated
        :return: The new node with LogSoftmax operation applied on each element.
        
    """
def lstm_sequence(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs LSTMSequence operation.
    
        :param X: The input tensor. Shape: [batch_size, seq_length, input_size].
        :param initial_hidden_state:    The hidden state tensor.
                                        Shape: [batch_size, num_directions, hidden_size].
        :param initial_cell_state:      The cell state tensor.
                                        Shape: [batch_size, num_directions, hidden_size].
        :param sequence_lengths:        Specifies real sequence lengths for each batch element.
                                        Shape: [batch_size]. Integer type.
        :param W: Tensor with weights for matrix multiplication operation with input portion of data.
                  Expected format: fico
                  Shape: [num_directions, 4*hidden_size, input_size].
        :param R: The tensor with weights for matrix multiplication operation with hidden state.
                  Expected format: fico
                  Shape: [num_directions, 4*hidden_size, hidden_size].
        :param B: The sum of biases (weight and recurrence). Expected format: fico
                  Shape: [num_directions, 4*hidden_size].
        :param hidden_size: Specifies hidden state size.
        :param direction: Specifies if the RNN is forward, reverse, or bidirectional.
        :param activations: The list of three activation functions for gates.
        :param activations_alpha: The list of alpha parameters for activation functions.
        :param activations_beta: The list of beta parameters for activation functions.
        :param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
        :param name: An optional name of the output node.
    
        :return: The new node represents LSTMSequence. Node outputs count: 3.
        
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
        :param soft_nms_sigma: Tensor specifying the sigma parameter for Soft-NMS.
        :param box_encoding: Format of boxes data encoding.
        :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                       boxes across batches or not.
        :param output_type: Output element type.
        :return: The new node which performs NonMaxSuppression
        
    """
def rnn_sequence(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs RNNSequence operation.
    
        :param X: The input tensor. Shape: [batch_size, seq_length, input_size].
        :param initial_hidden_state:    The hidden state tensor.
                                        Shape: [batch_size, num_directions, hidden_size].
        :param sequence_lengths:        Specifies real sequence lengths for each batch element.
                                        Shape: [batch_size]. Integer type.
        :param W: Tensor with weights for matrix multiplication operation with input portion of data.
                  Shape: [num_directions, hidden_size, input_size].
        :param R: The tensor with weights for matrix multiplication operation with hidden state.
                  Shape: [num_directions, hidden_size, hidden_size].
        :param B: The sum of biases (weight and recurrence).
                  Shape: [num_directions, hidden_size].
        :param hidden_size: Specifies hidden state size.
        :param direction: Specifies if the RNN is forward, reverse, or bidirectional.
        :param activations: The list of three activation functions for gates.
        :param activations_alpha: The list of alpha parameters for activation functions.
        :param activations_beta: The list of beta parameters for activation functions.
        :param clip: Specifies bound values [-C, C] for tensor clipping performed before activations.
        :param name: An optional name of the output node.
    
        :return: The new node represents RNNSequence. Node outputs count: 2.
        
    """
def round(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Apply Round operation on each element of input tensor.
    
        :param data: The tensor providing input data.
        :param mode: Rule to round halfway cases. If set to 'half_to_even' then halfs round to the nearest even
            integer or rounding in such a way that the result heads away from zero if `mode` attribute is
            'half_away_from_zero`.
        :param name: An optional name of the output node.
        :return: The new node with Round operation applied on each element.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset5: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset5')
