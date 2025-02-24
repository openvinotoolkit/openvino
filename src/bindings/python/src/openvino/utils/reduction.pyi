from __future__ import annotations
import openvino._pyopenvino
from openvino._pyopenvino import Node
__all__ = ['Node', 'get_reduction_axes']
def get_reduction_axes(node: openvino._pyopenvino.Node, reduction_axes: typing.Optional[typing.Iterable[int]]) -> typing.Iterable[int]:
    """
    Get reduction axes if it is None and convert it to set if its type is different.
    
        If reduction_axes is None we default to reduce all axes.
    
        :param node: The node we fill reduction axes for.
        :param reduction_axes: The collection of indices of axes to reduce. May be None.
    
        returns: Set filled with indices of axes we want to reduce.
        
    """
