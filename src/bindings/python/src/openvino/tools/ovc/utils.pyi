from __future__ import annotations
from importlib import metadata as importlib_metadata
from openvino_telemetry.backend import backend_ga4
from openvino.tools.ovc.error import Error
import importlib as importlib
import logging as log
import numpy
import numpy as np
import numpy.ma.core
import openvino_telemetry as tm
import os as os
import sys as sys
__all__ = ['Error', 'backend_ga4', 'bool_cast', 'check_values_equal', 'dynamic_dimension', 'get_ir_version', 'get_mo_root_dir', 'import_openvino_tokenizers', 'importlib', 'importlib_metadata', 'log', 'mo_array', 'np', 'np_map_cast', 'os', 'refer_to_faq_msg', 'sys', 'tm', 'validate_batch_in_shape']
def bool_cast(x):
    ...
def check_values_equal(val1, val2):
    ...
def get_ir_version():
    """
    
        Default IR version.
        :return: the IR version
        
    """
def get_mo_root_dir():
    """
    
        Return the absolute path to the Model Conversion API root directory (where mo folder is located)
        :return: path to the MO root directory
        
    """
def import_openvino_tokenizers():
    ...
def mo_array(value: typing.Union[typing.Iterable[typing.Union[float, int]], float, int], dtype = None) -> numpy.ndarray:
    """
    
        This function acts in a same way as np.array except for the case when dtype is not provided
        and np.array return fp64 array this function returns fp32 array
        
    """
def refer_to_faq_msg(question_num: int):
    ...
def validate_batch_in_shape(shape, layer_name: str):
    """
    
        Raises Error #39 if shape is not valid for setting batch size
        Parameters
        ----------
        shape: current shape of layer under validation
        layer_name: name of layer under validation
        
    """
dynamic_dimension: numpy.ma.core.MaskedConstant  # value = masked
np_map_cast: dict  # value = {bool: <function <lambda> at 0x7f582886fe20>, numpy.int8: <function <lambda> at 0x7f5828688360>, numpy.int16: <function <lambda> at 0x7f582868a200>, numpy.int32: <function <lambda> at 0x7f582868a2a0>, numpy.int64: <function <lambda> at 0x7f582868a340>, numpy.uint8: <function <lambda> at 0x7f582868a3e0>, numpy.uint16: <function <lambda> at 0x7f582868a480>, numpy.uint32: <function <lambda> at 0x7f582868a520>, numpy.uint64: <function <lambda> at 0x7f582868a5c0>, numpy.float16: <function <lambda> at 0x7f582868a660>, numpy.float32: <function <lambda> at 0x7f582868a700>, numpy.float64: <function <lambda> at 0x7f582868a7a0>, str: <function <lambda> at 0x7f582868a840>}
