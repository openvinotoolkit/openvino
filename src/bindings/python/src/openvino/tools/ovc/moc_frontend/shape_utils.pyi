# type: ignore
from __future__ import annotations
from openvino._pyopenvino import Dimension
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Shape
from openvino.tools.ovc.error import Error
import numpy as np
import openvino._pyopenvino
import sys as sys
__all__ = ['Dimension', 'Error', 'PartialShape', 'Shape', 'get_dynamic_dims', 'get_static_shape', 'is_shape_type', 'np', 'sys', 'tensor_to_int_list', 'to_partial_shape']
def get_dynamic_dims(shape: [openvino._pyopenvino.PartialShape, list, tuple]):
    ...
def get_static_shape(shape: [openvino._pyopenvino.PartialShape, list, tuple], dynamic_value = None):
    ...
def is_shape_type(value):
    ...
def tensor_to_int_list(tensor):
    ...
def to_partial_shape(shape):
    ...
