# type: ignore
"""
Factory functions for all openvino ops.
"""
from functools import partial
from __future__ import annotations
from openvino._pyopenvino import Node
from openvino.utils.decorators import nameable_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import as_node
from openvino.utils.types import as_nodes
from openvino.utils.types import make_constant_node
import functools
import openvino._pyopenvino
import typing
__all__ = ['Node', 'NodeInput', 'as_node', 'as_nodes', 'interpolate', 'is_finite', 'is_inf', 'is_nan', 'make_constant_node', 'nameable_op', 'partial', 'unique']
def interpolate(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perform interpolation of independent slices in input tensor.
    
        :param  image:         The node providing input tensor with data for interpolation.
        :param  output_shape:  1D tensor describing output shape for spatial axes.
        :param  scales:        1D tensor describing scales for spatial axes.
        :param  mode:          Specifies type of interpolation. Possible values are: nearest, linear,
                               linear_onnx, cubic.
        :param  shape_calculation_mode:
                               Specifies which input, sizes or scales, is used to calculate an output
                               shape.
        :param  pads_begin:    Specifies the number of pixels to add to the beginning of the image
                               being interpolated. Default is None.
        :param  pads_end:      Specifies the number of pixels to add to the end of the image being
                               interpolated. Default is None.
        :param  coordinate_transformation_mode:
                               Specifies how to transform the coordinate in the resized tensor to the
                               coordinate in the original tensor. Default is "half_pixel".
        :param  nearest_mode:  Specifies round mode when mode == nearest and is used only when
                               mode == nearest. Default is "round_prefer_floor".
        :param  antialias:     Specifies whether to perform anti-aliasing. Default is False.
        :param  cube_coeff:    Specifies the parameter a for cubic interpolation. Default is -0.75.
        :param  axes:          1D tensor specifying dimension indices where interpolation is applied.
                               Default is None.
        :param  name:          Optional name for the output node. Default is None.
        :return: Node representing interpolation operation.
        
    """
def is_finite(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Performs element-wise mapping from NaN and Infinity to False. Other values are mapped to True.
    
        :param  data:          A tensor of floating-point numeric type and arbitrary shape.
        :param  name:          Optional name for the output node. The default is None.
        :return: Node representing is_finite operation.
        
    """
def is_inf(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Return a node which performs IsInf operation.
    
        :param data: The input tensor.
        :param attributes: Optional dictionary containing IsInf attributes.
        :param name: Optional name of the node.
    
        Available attributes:
    
        * detect_negative   Specifies whether to map negative infinities to true in output map.
                            Range of values: true, false
                            Default value: true
                            Required: no
        * detect_positive   Specifies whether to map positive infinities to true in output map.
                            Range of values: true, false
                            Default value: true
                            Required: no
    
        :return: A new IsInf node.
        
    """
def is_nan(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Performs element-wise mapping from NaN to True. Other values are mapped to False.
    
        :param  data:          A tensor of floating point numeric type and arbitrary shape.
        :param  name:          Optional name for the output node. Default is None.
        :return: Node representing is_nan operation.
        
    """
def unique(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Operator which selects and returns unique elements or unique slices of the input tensor.
    
        :param  data:               Input data tensor.
        :param  axis:               (Optional) An input tensor containing the axis value.
                                    If not provided or None, data input is considered as a flattened tensor.
                                    Default value: None.
        :param  sorted:             (Optional) Controls the order of the returned unique values,
                                    sorts ascendingly when true.
                                    Default value: True.
        :param  index_element_type: (Optional) The data type set for outputs containing indices.
                                    Default value: "i64".
        :param  count_element_type: (Optional) The data type set for the output with repetition count.
                                    Default value: "i64".
        :param name:                (Optional) A name for the output node. Default value: None.
        :return: Node representing Unique operation.
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset10: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset10')
_get_node_factory_opset4: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset4')
