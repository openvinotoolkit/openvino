from __future__ import annotations
import logging as logging
import openvino._pyopenvino
from openvino._pyopenvino import AxisSet
import typing
__all__ = ['AxisSet', 'TensorShape', 'get_broadcast_axes', 'log', 'logging']
def get_broadcast_axes(output_shape: typing.List[int], input_shape: typing.List[int], axis: typing.Optional[int] = None) -> openvino._pyopenvino.AxisSet:
    """
    Generate a list of broadcast axes for openvino broadcast.
    
        Informally, a broadcast "adds" axes to the input tensor,
        replicating elements from the input tensor as needed to fill the new dimensions.
        Function calculate which of the output axes are added in this way.
    
        :param output_shape: The new shape for the output tensor.
        :param input_shape: The shape of input tensor.
        :param axis: The axis along which we want to replicate elements.
    
        returns: The indices of added axes.
        
    """
TensorShape: typing._GenericAlias  # value = typing.List[int]
log: logging.Logger  # value = <Logger openvino.utils.broadcasting (INFO)>
