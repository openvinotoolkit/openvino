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
import warnings as warnings
__all__ = ['Constant', 'Node', 'NodeFactory', 'NodeInput', 'NumericData', 'NumericType', 'Parameter', 'ScalarData', 'Shape', 'TensorShape', 'as_node', 'as_nodes', 'assert_list_of_ints', 'batch_to_space', 'binary_op', 'check_valid_attributes', 'gelu', 'get_dtype', 'get_element_type', 'get_element_type_str', 'is_non_negative_value', 'is_positive_value', 'make_constant_node', 'mvn', 'nameable_op', 'np', 'partial', 'reorg_yolo', 'roi_pooling', 'space_to_batch', 'unary_op', 'warnings']
def batch_to_space(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform BatchToSpace operation on the input tensor.
    
        BatchToSpace permutes data from the batch dimension of the data tensor into spatial dimensions.
    
        :param data: Node producing the data tensor.
        :param block_shape: The sizes of the block of values to be moved.
        :param crops_begin: Specifies the amount to crop from the beginning along each axis of `data`.
        :param crops_end: Specifies the amount to crop from the end along each axis of `data`.
        :param name: Optional output node name.
        :return: The new node performing a BatchToSpace operation.
        
    """
def gelu(input_value, *args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform Gaussian Error Linear Unit operation element-wise on data from input node.
    
        Computes GELU function:
    
        \\f[ f(x) = 0.5\\cdot x\\cdot(1 + erf( \\dfrac{x}{\\sqrt{2}}) \\f]
    
        For more information refer to [Gaussian Error Linear Unit (GELU)](https://arxiv.org/pdf/1606.08415.pdf>)
    
        :param node: Input tensor. One of: input node, array or scalar.
        :param name: Optional output node name.
        :return: The new node performing a GELU operation on its input data element-wise.
        
    """
def mvn(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform Mean Variance Normalization operation on data from input node.
    
        Computes MVN on the input tensor `data` (called `X`) using formula:
    
        \\f[ Y = \\dfrac{X-EX}{\\sqrt{E(X-EX)^2}} \\f]
    
        :param data: The node with data tensor.
        :param across_channels: Denotes if mean values are shared across channels.
        :param normalize_variance: Denotes whether to perform variance normalization.
        :param eps: The number added to the variance to avoid division by zero
                    when normalizing the value. Scalar value.
        :param name: Optional output node name.
        :return: The new node performing a MVN operation on input tensor.
        
    """
def reorg_yolo(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces the ReorgYolo operation.
    
        :param input:   Input data.
        :param stride:  Stride to reorganize input by.
        :param name:    Optional name for output node.
        :return: ReorgYolo node.
        
    """
def roi_pooling(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which produces an ROIPooling operation.
    
        :param input:          Input feature map `{N, C, ...}`.
        :param coords:         Coordinates of bounding boxes.
        :param output_roi:     Height/Width of ROI output features (shape).
        :param spatial_scale:  Ratio of input feature map over input image size (float).
        :param method:         Method of pooling - string: "max" or "bilinear". Default: "max"
        :param output_size:    (DEPRECATED!) Height/Width of ROI output features (shape).
                               Will override `output_roi` if used and change behavior of the operator.
        :return:               ROIPooling node.
        
    """
def space_to_batch(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform SpaceToBatch operation on the input tensor.
    
        SpaceToBatch permutes data tensor blocks of spatial data into batch dimension.
        The operator returns a copy of the input tensor where values from spatial blocks dimensions
        are moved in the batch dimension
    
        :param data: Node producing the data tensor.
        :param block_shape: The sizes of the block of values to be moved.
        :param pads_begin: Specifies the padding for the beginning along each axis of `data`.
        :param pads_end: Specifies the padding for the ending along each axis of `data`.
        :param name: Optional output node name.
        :return: The new node performing a SpaceToBatch operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
NumericData: typing._UnionGenericAlias  # value = typing.Union[int, float, numpy.ndarray]
NumericType: typing._UnionGenericAlias  # value = typing.Union[type, numpy.dtype]
ScalarData: typing._UnionGenericAlias  # value = typing.Union[int, float]
TensorShape: typing._GenericAlias  # value = typing.List[int]
_get_node_factory_opset2: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset2')
