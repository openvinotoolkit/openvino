# type: ignore
from __future__ import annotations
from openvino.frontend.frontend import FrontEnd
from openvino._pyopenvino import InputModel
from openvino._pyopenvino import NotImplementedFailure
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Place
from openvino._pyopenvino import Type
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.moc_frontend.analysis import json_model_analysis_dump
from openvino.tools.ovc.moc_frontend.extractor import convert_params_lists_to_dicts
from openvino.tools.ovc.moc_frontend.extractor import fe_output_user_data_repack
from openvino.tools.ovc.moc_frontend.extractor import fe_user_data_repack
from openvino.tools.ovc.utils import mo_array
from openvino.utils.types import get_element_type
from openvino.utils.types import get_numpy_ctype
import argparse as argparse
import logging as log
import numpy
import numpy as np
import openvino.frontend.frontend
import os as os
import sys as sys
__all__ = ['Error', 'FrontEnd', 'InputModel', 'NotImplementedFailure', 'PartialShape', 'Place', 'Type', 'argparse', 'convert_params_lists_to_dicts', 'fe_output_user_data_repack', 'fe_user_data_repack', 'get_element_type', 'get_enabled_and_disabled_transforms', 'get_numpy_ctype', 'json_model_analysis_dump', 'log', 'mo_array', 'moc_pipeline', 'np', 'np_map_cast', 'os', 'raise_exception_for_input_output_cut', 'sys']
def get_enabled_and_disabled_transforms():
    """
    
        :return: tuple of lists with force enabled and disabled id of transformations.
        
    """
def moc_pipeline(argv: argparse.Namespace, moc_front_end: openvino.frontend.frontend.FrontEnd):
    """
    
        Load input model and convert it to nGraph function
        :param: argv: parsed command line arguments
        :param: moc_front_end: Loaded Frontend for converting input model
        :return: converted nGraph function ready for serialization
        
    """
def raise_exception_for_input_output_cut(model_inputs_or_outputs: typing.List[openvino._pyopenvino.Place], new_nodes: typing.List[dict], is_input: bool):
    ...
np_map_cast: dict  # value = {bool: <function <lambda> at memory_address>, numpy.int8: <function <lambda> at memory_address>, numpy.int16: <function <lambda> at memory_address>, numpy.int32: <function <lambda> at memory_address>, numpy.int64: <function <lambda> at memory_address>, numpy.uint8: <function <lambda> at memory_address>, numpy.uint16: <function <lambda> at memory_address>, numpy.uint32: <function <lambda> at memory_address>, numpy.uint64: <function <lambda> at memory_address>, numpy.float16: <function <lambda> at memory_address>, numpy.float32: <function <lambda> at memory_address>, numpy.float64: <function <lambda> at memory_address>, str: <function <lambda> at memory_address>}
