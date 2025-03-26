# type: ignore
"""
Factory functions for ops added to openvino opset13.
"""
from functools import partial
from functools import singledispatch
from __future__ import annotations
from openvino.opset1.ops import convert_like
from openvino._pyopenvino import Node
from openvino._pyopenvino import Output
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Tensor
from openvino._pyopenvino import Type
from openvino._pyopenvino.op import Constant
from openvino._pyopenvino.op import Result
from openvino.utils.decorators import binary_op
from openvino.utils.decorators import nameable_op
from openvino.utils.decorators import overloading
from openvino.utils.decorators import unary_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
import functools
import logging as logging
import numpy as np
import openvino._pyopenvino
import openvino.utils.decorators
import typing
__all__ = ['Constant', 'Node', 'NodeInput', 'NumericData', 'NumericType', 'Output', 'Result', 'Shape', 'Tensor', 'Type', 'as_node', 'as_nodes', 'binary_op', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor', 'constant', 'convert_like', 'fake_convert', 'fake_quantize', 'log', 'logging', 'multinomial', 'nameable_op', 'nms_rotated', 'np', 'overloading', 'partial', 'result', 'scaled_dot_product_attention', 'singledispatch', 'unary_op']
def bitwise_and(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs bitwise AND operation on input nodes element-wise.
    
        For boolean input tensors, operator is equivalent to logical_and.
    
        :param left_node: Tensor of integer or boolean datatype providing data.
        :param right_node: Tensor of integer or boolean datatype providing data.
        :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
        :param name: The optional new name for output node.
        :return: The node performing bitwise AND operation on input nodes corresponding elements.
        
    """
def bitwise_not(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs bitwise NOT operation on input node element-wise.
    
        For boolean input tensors, operator is equivalent to logical_not.
    
        :param node: Tensor of integer or boolean datatype providing data.
        :param name: The optional new name for output node.
        :return: The node performing bitwise NOT operation on the given tensor.
        
    """
def bitwise_or(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs bitwise OR operation on input nodes element-wise.
    
        For boolean input tensors, operator is equivalent to logical_or.
    
        :param left_node: Tensor of integer or boolean datatype providing data.
        :param right_node: Tensor of integer or boolean datatype providing data.
        :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
        :param name: The optional new name for output node.
        :return: The node performing bitwise OR operation on input nodes corresponding elements.
        
    """
def bitwise_xor(left, right, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return node which performs bitwise XOR operation on input nodes element-wise.
    
        For boolean input tensors, operator is equivalent to logical_xor.
    
        :param left_node: Tensor of integer or boolean datatype providing data.
        :param right_node: Tensor of integer or boolean datatype providing data.
        :param auto_broadcast: The type of broadcasting specifies rules used for auto-broadcasting of input tensors. Defaults to “NUMPY”.
        :param name: The optional new name for output node.
        :return: The node performing bitwise XOR operation on input nodes corresponding elements.
        
    """
def fake_convert(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs FakeConvert.
    
        FakeConvert is experimental and may change in the future.
        .. warning:: FakeConvert is experimental and may change in the future.
    
        :param data: The node with data tensor with FP16, BF16 or FP32 datatype.
        :param scale: Tensor with a scale factor for the data input value,
                      of the same type as the data, and shape Numpy-broadcastable to data.
        :param shift: Optional tensor with value to subtract before and add after conversion of the data input value,
                      of the same type as the data, and shape Numpy-broadcastable to data.
        :param destination_type: Type to emulate, string of either "f8e4m3" or "f8e5m2".
        :param name: The optional new name for output node.
    
        :return: The new node performing FakeConvert operation.
        
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
        :param name:           Optional name of the new node.
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
def multinomial(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which generates a sequence of class indices sampled from the multinomial distribution.
    
        :param probs: Tensor with probabilities of floating-point type, and shape [batch_size, class_size].
        :param num_samples: Tensor (scalar or 1D) a single element of type i32 or i64,
                            specifying the number of samples to draw from the multinomial distribution.
        :param convert_type: Specifies the output tensor type, possible values: 'i64', 'i32'.
        :param with_replacement: Flag that specifies whether to sample with replacement.
        :param log_probs: Flag that specifies whether *probs* should be treated as unnormalized log probabilities.
        :param global_seed: Specifies global seed value. Required to be a positive integer or 0.
        :param op_seed: Specifies operational seed value. Required to be a positive integer or 0.
        :param name: The optional new name for output node.
    
        :return: The new node performing Multinomial operation.
        
    """
def nms_rotated(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs NMSRotated.
    
        :param boxes: Tensor with box coordinates of floating point type and shape [num_batches, num_boxes, 5],
                      where the last dimension is defined as [x_ctr, y_ctr, width, height, angle_radians].
        :param scores: Tensor with box scores of floating point type and shape [num_batches, num_classes, num_boxes].
        :param max_output_boxes_per_class: Tensor (scalar or 1D) of integer type, specifying maximum number of boxes
                                            to be selected per class.
        :param iou_threshold: Tensor (scalar or 1D) of floating point type, specifying intersection over union threshold
        :param score_threshold: Tensor (scalar or 1D) of floating point type, specifying minimum score to consider box for the processing.
        :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                       boxes across batches or not.
        :param output_type: Output element type.
        :param clockwise: Flag that specifies direction of the box rotation.
        :return: The new node which performs NMSRotated
        
    """
def result(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which represents an output of a graph (Model).
    
        :param data: The tensor containing the input data
        :return: Result node
        
    """
def scaled_dot_product_attention(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which implements Scaled Dot Product Attention.
    
        :param query: Query tensor of shape [N, ..., L, E] and floating-point datatype.
        :param key: Key tensor of shape [N, ..., S, E] and floating-point datatype.
        :param value: Value tensor of shape [N, ..., S, Ev] and floating-point datatype.
        :param attention_mask: Optional attention mask tensor of shape [N, ..., L, S] or scalar float type zero value.
                               Refer to the operation specification for a complete description.
        :param scale: Optional alternative scale, a floating-point type scalar.
        :param causal: If true, then autogenerates causal attention mask instead of using attention_mask input.
                       In this case attention_mask input is ignored.
        :param name: The optional new name for output node.
    
        :return: The new node performing Scaled Dot Product Attention operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
_get_node_factory_opset13: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset13')
constant: openvino.utils.decorators.MultiMethod  # value = <openvino.utils.decorators.MultiMethod object>
log: logging.Logger  # value = <Logger openvino.opset13.ops (INFO)>
