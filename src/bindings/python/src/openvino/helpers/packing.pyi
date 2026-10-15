# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Shape
from openvino._pyopenvino import Type
import numpy
import numpy as np
import openvino._pyopenvino
__all__: list[str] = ['Shape', 'Type', 'np', 'pack_data', 'unpack_data']
def _pack_u3(array: numpy.ndarray) -> numpy.ndarray:
    """
    Pack u3 values using transposed packing scheme.
    
        8 values (each 3 bits) are packed into 3 bytes:
        - Byte 0: bits [1:0] of values 0-3 (4 values * 2 bits = 8 bits)
        - Byte 1: bits [1:0] of values 4-7 (4 values * 2 bits = 8 bits)
        - Byte 2: bits [2] of all 8 values (8 values * 1 bit = 8 bits)
        
    """
def _pack_u6(array: numpy.ndarray) -> numpy.ndarray:
    """
    Pack u6 values using transposed packing scheme.
    
        4 values (each 6 bits) are packed into 3 bytes:
        - Byte 0: bits [3:0] of values 0-1 (2 values * 4 bits = 8 bits)
        - Byte 1: bits [3:0] of values 2-3 (2 values * 4 bits = 8 bits)
        - Byte 2: bits [5:4] of all 4 values (4 values * 2 bits = 8 bits)
        
    """
def _unpack_u3(array: numpy.ndarray, shape: typing.Union[list, openvino._pyopenvino.Shape]) -> numpy.ndarray:
    """
    Unpack u3 values using transposed unpacking scheme.
        
        3 bytes are unpacked into 8 values (each 3 bits).
        
    """
def _unpack_u6(array: numpy.ndarray, shape: typing.Union[list, openvino._pyopenvino.Shape]) -> numpy.ndarray:
    """
    Unpack u6 values using transposed unpacking scheme.
        
        3 bytes are unpacked into 4 values (each 6 bits).
        
    """
def pack_data(array: numpy.ndarray, type: openvino._pyopenvino.Type) -> numpy.ndarray:
    """
    Represent array values as u1, u2, u3, u4, u6 or i4 openvino element type and pack them into uint8 numpy array.
    
        For u1, u4, i4: Standard bit packing where 8 % bitwidth == 0
        For u3: Transposed packing - 8 values in 3 bytes
        For u6: Transposed packing - 4 values in 3 bytes
    
        If the number of elements in array is odd we pad them with zero value to be able to fit the bit
        sequence into the uint8 array.
    
        Example: two uint8 values - [7, 8] can be represented as uint4 values and be packed into one int8
                 value - [120], because [7, 8] bit representation is [0111, 1000] will be viewed
                 as [01111000], which is bit representation of [120].
    
        :param array: numpy array with values to pack.
        :type array: numpy array
        :param type: Type to interpret the array values. Type must be u1, u2, u3, u4, u6, i4, nf4 or f4e2m1.
        :type type: openvino.Type
        
    """
def unpack_data(array: numpy.ndarray, type: openvino._pyopenvino.Type, shape: typing.Union[list, openvino._pyopenvino.Shape]) -> numpy.ndarray:
    """
    Extract openvino element type values from array into new uint8/int8 array given shape.
    
        For u1, u4, i4: Standard bit unpacking where 8 % bitwidth == 0
        For u3: Transposed unpacking - 8 values from 3 bytes
        For u6: Transposed unpacking - 4 values from 3 bytes
    
        Example: uint8 value [120] can be represented as two u4 values and be unpacked into [7, 8]
                 because [120] bit representation is [01111000] will be viewed as [0111, 1000],
                 which is bit representation of [7, 8].
    
        :param array: numpy array to unpack.
        :type array: numpy array
        :param type: Type to extract from array values. Type must be u1, u2, u3, u4, u6, i4, nf4 or f4e2m1.
        :type type: openvino.Type
        :param shape: the new shape for the unpacked array.
        :type shape: Union[list, openvino.Shape]
        
    """
