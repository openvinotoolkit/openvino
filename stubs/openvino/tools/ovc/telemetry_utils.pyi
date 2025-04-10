# type: ignore
from __future__ import annotations
from openvino._pyopenvino import get_version as get_rt_version
from openvino_telemetry.backend import backend_ga4
from openvino.tools.ovc.cli_parser import get_params_with_paths_list
from openvino.tools.ovc.utils import check_values_equal
import argparse as argparse
import numbers as numbers
import openvino_telemetry as tm
import os as os
__all__ = ['arg_to_str', 'argparse', 'backend_ga4', 'check_values_equal', 'get_params_with_paths_list', 'get_rt_version', 'get_tid', 'init_ovc_telemetry', 'init_telemetry_class', 'is_keras3', 'is_optimum', 'is_torch_compile', 'numbers', 'os', 'send_conversion_result', 'send_framework_info', 'send_params_info', 'telemetry_params', 'tm']
def arg_to_str(arg):
    ...
def get_tid():
    """
    
        This function returns the ID of the database to send telemetry.
        
    """
def init_ovc_telemetry(app_name = 'OVC', app_version = None):
    ...
def init_telemetry_class(tid, app_name, app_version, backend, enable_opt_in_dialog, disable_in_ci):
    ...
def is_keras3():
    ...
def is_optimum():
    ...
def is_torch_compile():
    ...
def send_conversion_result(conversion_result: str, need_shutdown = False):
    ...
def send_framework_info(framework: str):
    """
    
        This function sends information about used framework.
        :param framework: framework name.
        
    """
def send_params_info(params: dict):
    """
    
        This function sends information about used command line parameters.
        :param params: command-line parameters dictionary.
        
    """
telemetry_params: dict = {'TID': 'G-W5E9RNLD4H'}
