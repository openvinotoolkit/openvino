# type: ignore
from collections import namedtuple
from collections import OrderedDict
from __future__ import annotations
from openvino._pyopenvino import Dimension
from openvino._pyopenvino import PartialShape
from openvino._pyopenvino import Type
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.help import get_convert_model_help_specifics
from openvino.tools.ovc.moc_frontend.shape_utils import is_shape_type
from openvino.tools.ovc.moc_frontend.shape_utils import to_partial_shape
from openvino.tools.ovc.moc_frontend.type_utils import is_type
from openvino.tools.ovc.moc_frontend.type_utils import to_ov_type
from openvino.tools.ovc.utils import get_mo_root_dir
import argparse as argparse
import inspect as inspect
import openvino as openvino
import os as os
import pathlib as pathlib
import re as re
__all__ = ['CanonicalizePathCheckExistenceAction', 'Dimension', 'Error', 'Formatter', 'OrderedDict', 'ParamDescription', 'PartialShape', 'Type', 'add_args_by_description', 'argparse', 'canonicalize_and_check_paths', 'check_bool', 'depersonalize', 'get_absolute_path', 'get_all_cli_parser', 'get_available_front_ends', 'get_common_cli_options', 'get_common_cli_parser', 'get_convert_model_help_specifics', 'get_mo_convert_params', 'get_mo_root_dir', 'get_model_name', 'get_model_name_from_args', 'get_node_name_with_port_from_input_value', 'get_params_with_paths_list', 'get_shape_from_input_value', 'input_model_details', 'input_to_input_cut_info', 'inspect', 'is_shape_type', 'is_single_input', 'is_type', 'namedtuple', 'openvino', 'os', 'parse_input_value', 'parse_inputs', 'pathlib', 're', 'readable_dirs_or_files_or_empty', 'readable_file_or_dir_or_object', 'remove_shape_from_input_value', 'single_input_to_input_cut_info', 'split_inputs', 'to_ov_type', 'to_partial_shape']
class CanonicalizePathCheckExistenceAction(argparse.Action):
    """
    
        Expand user home directory paths and convert relative-paths to absolute and check specified file or directory
        existence.
        
    """
    @staticmethod
    def check_value(values: typing.Union[str, typing.List[str], NoneType], param_name, try_mo_root = False, check_existence = True) -> typing.List[str]:
        ...
    def __call__(self, parser, namespace, values, option_string = None):
        ...
class Formatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        ...
    def _get_default_metavar_for_optional(self, action):
        ...
def add_args_by_description(args_group, params_description):
    ...
def canonicalize_and_check_paths(values: typing.Union[str, typing.List[str], NoneType], param_name, try_mo_root = False, check_existence = True) -> typing.List[str]:
    ...
def check_bool(value):
    ...
def depersonalize(value: str, key: str):
    ...
def get_absolute_path(path_to_file: str) -> str:
    """
    
        Deduces absolute path of the file by a given path to the file
        Args:
            path_to_file: path to the file
    
        Returns:
            absolute path of the file
        
    """
def get_all_cli_parser():
    """
    
        Specifies cli arguments for Model Conversion
    
        Returns
        -------
            ArgumentParser instance
        
    """
def get_available_front_ends(fem = None):
    ...
def get_common_cli_options(argv, is_python_api_used):
    ...
def get_common_cli_parser(parser: argparse.ArgumentParser = None):
    ...
def get_mo_convert_params():
    ...
def get_model_name(path_input_model: str) -> str:
    """
    
        Deduces model name by a given path to the input model
        Args:
            path_input_model: path to the input model
    
        Returns:
            name of the output IR
        
    """
def get_model_name_from_args(argv: argparse.Namespace):
    ...
def get_node_name_with_port_from_input_value(input_value: str):
    """
    
        Returns the node name (optionally with input/output port) from the input value
        :param input_value: string passed as input to the "input" command line parameter
        :return: the corresponding node name with input/output port
        
    """
def get_params_with_paths_list():
    ...
def get_shape_from_input_value(input_value: str):
    """
    
        Returns PartialShape corresponding to the shape specified in the input value string
        :param input_value: string passed as input to the "input" command line parameter
        :return: the corresponding shape and None if the shape is not specified in the input value
        
    """
def input_model_details(model):
    ...
def input_to_input_cut_info(input: [dict, tuple, list]):
    """
    
        Parses 'input' to list of InputCutInfo.
        :param input: input cut parameters passed by user
        :return: list of InputCutInfo with input cut parameters
        
    """
def is_single_input(input: [tuple, list]):
    """
    
        Checks if input has parameters for single input.
        :param input: list or tuple of input parameters or input shape or input name.
        :return: True if input has parameters for single input, otherwise False.
        
    """
def parse_input_value(input_value: str):
    """
    
        Parses a value of the "input" command line parameter and gets a node name, shape and value.
        The node name includes a port if it is specified.
        Shape and value is equal to None if they are not specified.
        Parameters
        ----------
        input_value
            string with a specified node name and shape.
            E.g. 'node_name:0[4]'
    
        Returns
        -------
            Node name, shape, value, data type
            E.g. 'node_name:0', '4', [1.0 2.0 3.0 4.0], np.float32
        
    """
def parse_inputs(inputs: str):
    ...
def readable_dirs_or_files_or_empty(paths: [str, list, tuple]):
    """
    
        Checks that comma separated list of paths are readable directories, files or a provided path is empty.
        :param paths: comma separated list of paths.
        :return: comma separated list of paths.
        
    """
def readable_file_or_dir_or_object(path: str):
    """
    
        Check that specified path is a readable file or directory.
        :param path: path to check
        :return: path if the file/directory is readable
        
    """
def remove_shape_from_input_value(input_value: str):
    """
    
        Removes the shape specification from the input string. The shape specification is a string enclosed with square
        brackets.
        :param input_value: string passed as input to the "input" command line parameter
        :return: string without shape specification
        
    """
def single_input_to_input_cut_info(input: [str, tuple, list, openvino._pyopenvino.PartialShape, openvino._pyopenvino.Type, type]):
    """
    
        Parses parameters of single input to InputCutInfo.
        :param input: input cut parameters of single input
        :return: InputCutInfo
        
    """
def split_inputs(input_str):
    ...
ParamDescription = ParamData
_InputCutInfo = InputCutInfo
