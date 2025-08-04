# type: ignore
from __future__ import annotations
from collections import OrderedDict
from collections.abc import Callable
from collections.abc import Iterable
from openvino._pyopenvino import OpConversionFailure
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import TelemetryExtension
from openvino._pyopenvino.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_0 import get_version as get_rt_version
from openvino.frontend.frontend import FrontEndManager
from openvino.frontend.tensorflow.utils import create_tf_graph_iterator
from openvino.frontend.tensorflow.utils import extract_model_graph
from openvino.frontend.tensorflow.utils import type_supported_by_tf_fe
from openvino.tools.ovc.cli_parser import depersonalize
from openvino.tools.ovc.cli_parser import get_available_front_ends
from openvino.tools.ovc.cli_parser import get_common_cli_options
from openvino.tools.ovc.cli_parser import get_mo_convert_params
from openvino.tools.ovc.cli_parser import input_to_input_cut_info
from openvino.tools.ovc.cli_parser import parse_inputs
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.error import FrameworkError
from openvino.tools.ovc.get_ov_update_message import get_compression_message
from openvino.tools.ovc.help import get_convert_model_help_specifics
from openvino.tools.ovc.logger import init_logger
from openvino.tools.ovc.moc_frontend.check_config import any_extensions_used
from openvino.tools.ovc.moc_frontend.jax_frontend_utils import get_jax_decoder
from openvino.tools.ovc.moc_frontend.moc_emit_ir import moc_emit_ir
from openvino.tools.ovc.moc_frontend.paddle_frontend_utils import paddle_frontend_converter
from openvino.tools.ovc.moc_frontend.pipeline import moc_pipeline
from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import extract_input_info_from_example
from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import get_pytorch_decoder
from openvino.tools.ovc.moc_frontend.pytorch_frontend_utils import get_pytorch_decoder_for_model_on_disk
from openvino.tools.ovc.moc_frontend.type_utils import to_ov_type
from openvino.tools.ovc.telemetry_utils import init_ovc_telemetry
from openvino.tools.ovc.telemetry_utils import send_conversion_result
from openvino.tools.ovc.telemetry_utils import send_params_info
from openvino.tools.ovc.utils import check_values_equal
from openvino.tools.ovc.version import VersionChecker
from openvino_telemetry.backend import backend_ga4
from pathlib import Path
import argparse as argparse
import collections.abc
import datetime as datetime
import logging as log
import openvino_telemetry as tm
import os as os
import sys as sys
import traceback as traceback
import tracemalloc as tracemalloc
__all__ = ['Callable', 'Error', 'FrameworkError', 'FrontEndManager', 'Iterable', 'OpConversionFailure', 'OrderedDict', 'PartialShape', 'Path', 'TelemetryExtension', 'VersionChecker', 'add_line_breaks', 'any_extensions_used', 'argparse', 'args_to_argv', 'arguments_post_parsing', 'backend_ga4', 'check_iterable', 'check_model_object', 'check_values_equal', 'create_tf_graph_iterator', 'datetime', 'depersonalize', 'driver', 'extract_input_info_from_example', 'extract_model_graph', 'filtered_extensions', 'get_available_front_ends', 'get_common_cli_options', 'get_compression_message', 'get_convert_model_help_specifics', 'get_jax_decoder', 'get_mo_convert_params', 'get_moc_frontends', 'get_non_default_params', 'get_pytorch_decoder', 'get_pytorch_decoder_for_model_on_disk', 'get_rt_version', 'init_logger', 'init_ovc_telemetry', 'input_model_is_object', 'input_to_input_cut_info', 'is_verbose', 'log', 'moc_emit_ir', 'moc_pipeline', 'normalize_inputs', 'os', 'pack_params_to_args_namespace', 'paddle_frontend_converter', 'parse_inputs', 'prepare_ir', 'print_argv', 'replace_ext', 'send_conversion_result', 'send_params_info', 'show_mo_convert_help', 'sys', 'tf_frontend_with_python_bindings_installed', 'tm', 'to_ov_type', 'traceback', 'tracemalloc', 'type_supported_by_tf_fe']
def _convert(cli_parser: argparse.ArgumentParser, args, python_api_used):
    ...
def add_line_breaks(text: str, char_num: int, line_break: str):
    ...
def args_to_argv(**kwargs):
    ...
def arguments_post_parsing(argv: argparse.Namespace):
    ...
def check_iterable(iterable: collections.abc.Iterable, func: collections.abc.Callable):
    ...
def check_model_object(argv):
    ...
def driver(argv: argparse.Namespace, non_default_params: dict):
    ...
def filtered_extensions(extensions):
    ...
def get_moc_frontends(argv: argparse.Namespace):
    ...
def get_non_default_params(argv, cli_parser):
    ...
def input_model_is_object(input_model):
    ...
def is_verbose(argv, args = None):
    ...
def normalize_inputs(argv: argparse.Namespace):
    """
    
        repacks params passed to convert_model and wraps resulting values into dictionaries or lists.
        After working of this method following values are set in argv:
    
        argv.input, argv.inputs_list - list of input names. Both values are used in some parts of MO.
        Could be good to refactor it and use only one of these values.
    
        argv.placeholder_shapes - dictionary where key is node name, value is PartialShape,
        or list of PartialShape if node names were not set.
    
        argv.placeholder_data_types - dictionary where key is node name, value is node np.type,
        or list of np.types if node names were not set.
    
        :param argv: OVC arguments
        
    """
def pack_params_to_args_namespace(args: dict, cli_parser: argparse.ArgumentParser, python_api_used):
    ...
def prepare_ir(argv: argparse.Namespace):
    ...
def print_argv(argv: argparse.Namespace):
    ...
def replace_ext(name: str, old: str, new: str):
    ...
def show_mo_convert_help():
    ...
