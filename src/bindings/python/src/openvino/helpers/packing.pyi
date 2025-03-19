# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Type
import numpy
import numpy as np
import openvino._pyopenvino
__all__ = ['Shape', 'Type', 'np', 'pack_data', 'unpack_data']
def pack_data(array: numpy.ndarray, type: openvino._pyopenvino.Type) -> numpy.ndarray:
    """
    Represent array values as u1,u4 or i4 openvino element type and pack them into uint8 numpy array.
    
        If the number of elements in array is odd we pad them with zero value to be able to fit the bit
        sequence into the uint8 array.
    
        Example: two uint8 values - [7, 8] can be represented as uint4 values and be packed into one int8
                 value - [120], because [7, 8] bit representation is [0111, 1000] will be viewed
                 as [01111000], which is bit representation of [120].
    
        :param array: numpy array with values to pack.
        :type array: numpy array
        :param type: Type to interpret the array values. Type must be u1, u4, i4, nf4 or f4e2m1.
        :type type: openvino.Type
        
    """
def unpack_data(array: numpy.ndarray, type: openvino._pyopenvino.Type, shape: typing.Union[list, openvino._pyopenvino.Shape]) -> numpy.ndarray:
    """
    Extract openvino element type values from array into new uint8/int8 array given shape.
    
        Example: uint8 value [120] can be represented as two u4 values and be unpacked into [7, 8]
                 because [120] bit representation is [01111000] will be viewed as [0111, 1000],
                 which is bit representation of [7, 8].
    
        :param array: numpy array to unpack.
        :type array: numpy array
        :param type: Type to extract from array values. Type must be u1, u4, i4, nf4 or f4e2m1.
        :type type: openvino.Type
        :param shape: the new shape for the unpacked array.
        :type shape: Union[list, openvino.Shape]
        
    """
