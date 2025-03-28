# type: ignore
"""
Factory functions for all openvino ops.
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
__all__ = ['Node', 'NodeInput', 'as_nodes', 'interpolate', 'nameable_op', 'partial', 'topk']
def interpolate(*args, **kwargs) -> openvino._pyopenvino.Node:
    """
    Perfors the interpolation of the input tensor.
    
        :param  image:         The node providing input tensor with data for interpolation.
        :param  scales_or_sizes:
                               1D tensor providing information used to calculate the output shape
                               of the operation. It might contain floats (scales) or integers(sizes).
        :param  mode:          Specifies type of interpolation. Possible values are: nearest, linear,
                               linear_onnx, cubic, bilinear_pillow, bicubic_pillow.
        :param  shape_calculation_mode:
                               Specifies how the scales_or_sizes input should be interpreted.
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
                               The default is None.
        :param  name:          Optional name for the output node. The default is None.
        :return: Node representing the interpolation operation.
        
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
        :param stable: Specifies whether the equivalent elements should maintain
                       their relative order from the input tensor during sorting.
        :return: The new node which performs TopK
        
    """
NodeInput: typing._UnionGenericAlias  # value = typing.Union[openvino._pyopenvino.Node, int, float, numpy.ndarray]
_get_node_factory_opset11: functools.partial  # value = functools.partial(<function _get_node_factory at memory_address>, 'opset11')
