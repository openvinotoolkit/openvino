# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import logging as log
import os
import re
from collections import OrderedDict, namedtuple
from distutils.util import strtobool
from itertools import zip_longest
from pathlib import Path
from operator import xor
from typing import List, Union
import numbers
import inspect

import numpy as np
from openvino.runtime import Layout, PartialShape, Dimension, Shape, Type # pylint: disable=no-name-in-module,import-error

import openvino
from openvino.tools.ovc.convert_data_type import destination_type_to_np_data_type
from openvino.tools.ovc.error import Error
from openvino.tools.ovc.utils import get_mo_root_dir
from openvino.tools.ovc.help import get_convert_model_help_specifics, get_to_string_methods_for_params


def extension_path_to_str_or_extensions_class(extension):
    if isinstance(extension, str):
        return extension
    elif isinstance(extension, Path):
        return str(extension)
    else:
        # Return unknown object as is.
        # The type of the object will be checked by frontend.add_extension() method
        return extension


def extensions_to_str_or_extensions_class(extensions):
    if extensions is None:
        return None
    extensions_list = []
    if isinstance(extensions, str):
        extensions_list = extensions.split(',')
    elif isinstance(extensions, list):
        for ext in extensions:
            ext = extension_path_to_str_or_extensions_class(ext)
            extensions_list.append(ext)
    else:
        extensions_list = [extension_path_to_str_or_extensions_class(extensions)]

    for ext in extensions_list:
        if isinstance(ext, str):
            readable_file_or_dir(ext)
    return extensions_list


def path_to_str(path):
    if path is None:
        return None
    if isinstance(path, str):
        return path
    elif isinstance(path, Path):
        return str(path)
    else:
        raise Exception("Incorrect type of {} expected str or Path, got {}".format(path, type(path)))


def path_to_str_or_object(value):
    if value is None or isinstance(value, str):
        return value
    elif isinstance(value, Path):
        return str(value)
    else:
        return value


def paths_to_str(paths):
    if paths is None:
        return None
    if isinstance(paths, list):
        paths_str = []
        for path in paths:
            paths_str.append(path_to_str(path))
        return ','.join(paths_str)
    else:
        path_to_str(paths)


def str_list_to_str(values):
    if values is None:
        return None
    if isinstance(values, str):
        return values
    elif isinstance(values, list):
        for value in values:
            if not isinstance(value, str):
                raise Error("Incorrect argument. {} expected to string, got type {}.".format(value, type(value)))
        return ','.join(values)
    else:
        raise Error("Incorrect argument. {} expected to string or list of strings, got type {}.".format(values, type(values)))


def is_shape_type(value):
    if isinstance(value, PartialShape):
        return True
    if isinstance(value, Shape):
        return True
    if isinstance(value, list) or isinstance(value, tuple):
        for dim in value:
            if not (isinstance(dim, Dimension) or isinstance(dim, int)):
                return False
        return True
    return False


def value_to_str(value, separator):
    if isinstance(value, np.ndarray):
        values = []
        for x in np.nditer(value):
            values.append(str(x))
        return "[" + separator.join(values) + "]"
    if isinstance(value, list):
        values = []
        for x in value:
            if not isinstance(x, numbers.Number):
                raise Exception("Incorrect value type. Expected numeric value, got {}".format(type(x)))
            values.append(str(x))
        return "[" + separator.join(values) + "]"
    if isinstance(value, bool):
        return "True" if value else "False"
    raise Exception("Incorrect value type. Expected np.ndarray or list, got {}".format(type(value)))


def single_input_to_input_cut_info(input: [str, tuple, list, PartialShape, Type, type]):
    """
    Parses parameters of single input to InputCutInfo.
    :param input: input cut parameters of single input
    :return: InputCutInfo
    """
    if isinstance(input, str):
        # Parse params from string
        node_name, shape, value, data_type = parse_input_value(input)
        # pylint: disable=no-member
        return openvino.tools.ovc.InputCutInfo(node_name,
                                              PartialShape(shape) if shape is not None else None,
                                              data_type,
                                              value)
    if isinstance(input, openvino.tools.ovc.InputCutInfo): # pylint: disable=no-member
        # Wrap input.shape to PartialShape if possible and wrap to InputCutInfo
        # pylint: disable=no-member
        return openvino.tools.ovc.InputCutInfo(input.name,
                                              PartialShape(input.shape) if input.shape is not None else None,
                                              input.type,
                                              input.value)
    if isinstance(input, (tuple, list, PartialShape)):
        # If input represents list with shape, wrap it to list. Single PartialShape also goes to this condition.
        # Check of all dimensions will be in is_shape_type(val) method below
        if len(input) > 0 and isinstance(input[0], (int, Dimension)):
            input = [input]

        # Check values of tuple or list and collect to InputCutInfo
        name = None
        inp_type = None
        shape = None
        for val in input:
            if isinstance(val, str):
                if name is not None:
                    raise Exception("More than one input name provided: {}".format(input))
                name = val
            elif isinstance(val, (type, Type)):
                if inp_type is not None:
                    raise Exception("More than one input type provided: {}".format(input))
                inp_type = val
            elif is_shape_type(val):
                if shape is not None:
                    raise Exception("More than one input shape provided: {}".format(input))
                shape = PartialShape(val)
            else:
                raise Exception("Incorrect input parameters provided. Expected tuple with input name, "
                                "input type or input shape. Got unknown object: {}".format(val))
        # pylint: disable=no-member
        return openvino.tools.ovc.InputCutInfo(name,
                                              PartialShape(shape) if shape is not None else None,
                                              inp_type,
                                              None)
    # Case when only type is set
    if isinstance(input, (type, Type)):
        return openvino.tools.ovc.InputCutInfo(None, None, input, None) # pylint: disable=no-member

    # We don't expect here single unnamed value. If list of int is set it is considered as shape.
    # Setting of value is expected only using InputCutInfo or string analog.

    raise Exception("Unexpected object provided for input. Expected openvino.tools.ovc.InputCutInfo "
                    "or tuple or str. Got {}".format(type(input)))


def input_to_input_cut_info(input: [str, tuple, list]):
    """
    Parses 'input' to list of InputCutInfo.
    :param input: input cut parameters passed by user
    :return: list of InputCutInfo with input cut parameters
    """
    if input is None:
        return []
    if isinstance(input, str):
        inputs = []
        # Split to list of string
        for input_value in split_inputs(input):

            # Parse string with parameters for single input
            node_name, shape, value, data_type = parse_input_value(input_value)
            # pylint: disable=no-member
            inputs.append(openvino.tools.ovc.InputCutInfo(node_name,
                                                         PartialShape(shape) if shape is not None else None,
                                                         data_type,
                                                         value))
        return inputs
    # pylint: disable=no-member
    if isinstance(input, openvino.tools.ovc.InputCutInfo):
        # Wrap to list and return
        return [input]
    if isinstance(input, tuple):
        # Case when input is single shape set in tuple
        if len(input) > 0 and isinstance(input[0], (int, Dimension)):
            input = [input]
        # Case when input is set as tuple. Expected that it is always single input.
        return [single_input_to_input_cut_info(input)]
    if isinstance(input, list):
        # Case when input is single shape set in list
        if len(input) > 0 and isinstance(input[0], (int, Dimension)):
            input = [input]
        inputs = []
        # Case when input is set as list. Expected that it is list of params for different inputs.
        for inp in input:
            inputs.append(single_input_to_input_cut_info(inp))
        return inputs
    # Case when single type or value is set, or unknown object
    return [single_input_to_input_cut_info(input)]


def freeze_placeholder_to_input_cut_info(inputs: list):
    """
    Parses freezing parts from input list.
    :param inputs: list of InputCutInfo with information from 'input' parameter
    :returns (placeholder_values, unnamed_placeholder_values), where
    placeholder_values - dictionary where key is node name, value is node value,
    unnamed_placeholder_values - list with unnamed node values
    """
    placeholder_values = {}
    unnamed_placeholder_values = []

    # Collect values for freezing from 'inputs'
    if inputs is not None and len(inputs) > 0:
        for input in inputs:
            node_name = input.name
            value = input.value
            if value is None:
                continue
            # Check for value conflict
            if node_name in placeholder_values and placeholder_values[node_name] != value:
                raise Error("Overriding replacement value of the placeholder with name '{}': old value = {}, new value = {}"
                            ".".format(node_name, placeholder_values[node_name], value))
            if node_name is not None:
                # Named input case, add to dictionary
                placeholder_values[node_name] = value
            else:
                # Unnamed input case, add to list
                unnamed_placeholder_values.append(value)

    return placeholder_values, unnamed_placeholder_values


def layout_to_str(layout):
    if isinstance(layout, str):
        return layout
    if isinstance(layout, Layout):
        return layout.to_string()
    raise Exception("Incorrect layout type. Expected Layout or string or dictionary, "
                    "where key is operation name and value is layout or list of layouts, got {}".format(type(layout)))

ParamDescription = namedtuple("ParamData",
                              ["description", "cli_tool_description", "to_string"])


def get_mo_convert_params():
    mo_convert_docs = openvino.tools.ovc.convert_model.__doc__ # pylint: disable=no-member
    mo_convert_params = {}
    group = "Framework-agnostic parameters:"    #FIXME: WA for unknown bug in this function
    mo_convert_params[group] = {}

    mo_convert_docs = mo_convert_docs[:mo_convert_docs.find('Returns:')]

    while len(mo_convert_docs) > 0:
        param_idx1 = mo_convert_docs.find(":param")
        if param_idx1 == -1:
            break
        param_idx2 = mo_convert_docs.find(":", param_idx1+1)
        param_name = mo_convert_docs[param_idx1+len(':param '):param_idx2]

        param_description_idx = mo_convert_docs.find(":param", param_idx2+1)
        param_description = mo_convert_docs[param_idx2+1: param_description_idx]

        group_name_idx = param_description.rfind('\n\n')
        group_name = ''
        if group_name_idx != -1:
            group_name = param_description[group_name_idx:].strip()

        param_description = param_description[:group_name_idx]
        param_description = param_description.strip()

        mo_convert_params[group][param_name] = ParamDescription(param_description, "", None)

        mo_convert_docs = mo_convert_docs[param_description_idx:]

        if group_name != '':
            mo_convert_params[group_name] = {}
            group = group_name

    # TODO: remove this when internal converting of params to string is removed <-- DO IT
    params_converted_to_string = get_to_string_methods_for_params()

    params_with_paths = get_params_with_paths_list()
    cli_tool_specific_descriptions = get_convert_model_help_specifics()

    for group_name, param_group in mo_convert_params.items():
        for param_name, d in param_group.items():
            to_str_method = None
            if param_name in params_converted_to_string:
                to_str_method = params_converted_to_string[param_name]
            elif param_name in params_with_paths:
                to_str_method = path_to_str

            cli_tool_description = None
            if param_name in cli_tool_specific_descriptions:
                cli_tool_description = cli_tool_specific_descriptions[param_name]

            desc = ParamDescription(d.description,
                                    cli_tool_description,
                                    to_str_method)
            mo_convert_params[group_name][param_name] = desc

    return mo_convert_params


class DeprecatedStoreTrue(argparse.Action):
    def __init__(self, nargs=0, **kw):
        super().__init__(nargs=nargs, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
        dep_msg = "Use of deprecated cli option {} detected. Option use in the following releases will be fatal. ".format(option_string)
        log.error(dep_msg, extra={'is_warning': True})
        setattr(namespace, self.dest, True)


class DeprecatedOptionCommon(argparse.Action):
    def __call__(self, parser, args, values, option_string):
        dep_msg = "Use of deprecated cli option {} detected. Option use in the following releases will be fatal. ".format(option_string)
        log.error(dep_msg, extra={'is_warning': True})
        setattr(args, self.dest, values)


class IgnoredAction(argparse.Action):
    def __init__(self, nargs=0, **kw):
        super().__init__(nargs=nargs, **kw)

    def __call__(self, parser, namespace, values, option_string=None):
        dep_msg = "Use of removed cli option '{}' detected. The option is ignored. ".format(option_string)
        log.error(dep_msg, extra={'is_warning': True})
        setattr(namespace, self.dest, True)


def canonicalize_and_check_paths(values: Union[str, List[str], None], param_name,
                                 try_mo_root=False, check_existence=True) -> List[str]:
    if values is not None:
        list_of_values = list()
        if isinstance(values, str):
            if values != "":
                list_of_values = values.split(',')
        elif isinstance(values, list):
            list_of_values = values
        else:
            raise Error('Unsupported type of command line parameter "{}" value'.format(param_name))

        if not check_existence:
            return [get_absolute_path(path) for path in list_of_values]

        for idx, val in enumerate(list_of_values):
            list_of_values[idx] = val

            error_msg = 'The value for command line parameter "{}" must be existing file/directory, ' \
                        'but "{}" does not exist.'.format(param_name, val)
            if os.path.exists(val):
                continue
            elif not try_mo_root or val == '':
                raise Error(error_msg)
            elif try_mo_root:
                path_from_mo_root = get_mo_root_dir() + '/ovc/' + val
                list_of_values[idx] = path_from_mo_root
                if not os.path.exists(path_from_mo_root):
                    raise Error(error_msg)

        return [get_absolute_path(path) for path in list_of_values]


class CanonicalizePathAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=False, check_existence=False)
        setattr(namespace, self.dest, list_of_paths)


class CanonicalizeTransformationPathCheckExistenceAction(argparse.Action):
    """
    Convert relative to the current and relative to mo root paths to absolute
    and check specified file or directory existence.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=True, check_existence=True)
        setattr(namespace, self.dest, ','.join(list_of_paths))


class CanonicalizePathCheckExistenceAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute and check specified file or directory
    existence.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=False, check_existence=True)
        setattr(namespace, self.dest, list_of_paths)


class CanonicalizeExtensionsPathCheckExistenceAction(argparse.Action):
    """
    Expand user home directory paths and convert relative-paths to absolute and check specified file or directory
    existence.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        list_of_paths = canonicalize_and_check_paths(values, param_name=option_string,
                                                     try_mo_root=False, check_existence=True)
        # Extensions paths are needed to be stored as list
        setattr(namespace, self.dest, list_of_paths)


class CanonicalizePathCheckExistenceIfNeededAction(CanonicalizePathCheckExistenceAction):

    def __call__(self, parser, namespace, values, option_string=None):
        if values is not None:
            if isinstance(values, str):
                if values != "":
                    super().__call__(parser, namespace, values, option_string)
                else:
                    setattr(namespace, self.dest, values)


class DeprecatedCanonicalizePathCheckExistenceAction(CanonicalizePathCheckExistenceAction):
    def __call__(self, parser, namespace, values, option_string=None):
        dep_msg = "Use of deprecated cli option {} detected. Option use in the following releases will be fatal. ".format(
            option_string)
        log.error(dep_msg, extra={'is_warning': True})
        super().__call__(parser, namespace, values, option_string)


def readable_file(path: str):
    """
    Check that specified path is a readable file.
    :param path: path to check
    :return: path if the file is readable
    """
    if not os.path.isfile(path):
        raise Error('The "{}" is not existing file'.format(path))
    elif not os.access(path, os.R_OK):
        raise Error('The "{}" is not readable'.format(path))
    else:
        return path


def readable_file_or_dir(path: str):
    """
    Check that specified path is a readable file or directory.
    :param path: path to check
    :return: path if the file/directory is readable
    """
    if not os.path.isfile(path) and not os.path.isdir(path):
        raise Error('The "{}" is not existing file or directory'.format(path))
    elif not os.access(path, os.R_OK):
        raise Error('The "{}" is not readable'.format(path))
    else:
        return path


def readable_dirs(paths: str):
    """
    Checks that comma separated list of paths are readable directories.
    :param paths: comma separated list of paths.
    :return: comma separated list of paths.
    """
    paths_list = [readable_dir(path) for path in paths.split(',')]
    return ','.join(paths_list)


def readable_dirs_or_empty(paths: str):
    """
    Checks that comma separated list of paths are readable directories of if it is empty.
    :param paths: comma separated list of paths.
    :return: comma separated list of paths.
    """
    if paths:
        return readable_dirs(paths)
    return paths


def readable_dirs_or_files_or_empty(paths: str):
    """
    Checks that comma separated list of paths are readable directories, files or a provided path is empty.
    :param paths: comma separated list of paths.
    :return: comma separated list of paths.
    """
    if paths:
        paths_list = [readable_file_or_dir(path) for path in paths.split(',')]
        return ','.join(paths_list)
    return paths


def readable_dir(path: str):
    """
    Check that specified path is a readable directory.
    :param path: path to check
    :return: path if the directory is readable
    """
    if not os.path.isdir(path):
        raise Error('The "{}" is not existing directory'.format(path))
    elif not os.access(path, os.R_OK):
        raise Error('The "{}" is not readable'.format(path))
    else:
        return path


def writable_dir(path: str):
    """
    Checks that specified directory is writable. The directory may not exist but it's parent or grandparent must exist.
    :param path: path to check that it is writable.
    :return: path if it is writable
    """
    if path is None:
        raise Error('The directory parameter is None')
    if os.path.exists(path):
        if os.path.isdir(path):
            if os.access(path, os.W_OK):
                return path
            else:
                raise Error('The directory "{}" is not writable'.format(path))
        else:
            raise Error('The "{}" is not a directory'.format(path))
    else:
        cur_path = path
        while os.path.dirname(cur_path) != cur_path:
            if os.path.exists(cur_path):
                break
            cur_path = os.path.dirname(cur_path)
        if cur_path == '':
            cur_path = os.path.curdir
        if os.access(cur_path, os.W_OK):
            return path
        else:
            raise Error('The directory "{}" is not writable'.format(cur_path))


def add_args_by_description(args_group, params_description):
    signature = inspect.signature(openvino.tools.ovc.convert_model) # pylint: disable=no-member
    filepath_args = get_params_with_paths_list()
    cli_tool_specific_descriptions = get_convert_model_help_specifics()
    for param_name, param_description in params_description.items():
        if param_name in ['share_weights', 'example_input']:
            continue
        if param_name == 'input_model':
            # input_model is not a normal key for a tool, it will collect all untagged keys
            cli_param_name = param_name
        else:
            cli_param_name = '--' + param_name
        if cli_param_name not in args_group._option_string_actions:
            # Get parameter specifics
            param_specifics = cli_tool_specific_descriptions[param_name] if param_name in \
                                                                            cli_tool_specific_descriptions else {}
            help_text = param_specifics['description'] if 'description' in param_specifics \
                else param_description.description
            action = param_specifics['action'] if 'action' in param_specifics else None
            param_type = param_specifics['type'] if 'type' in param_specifics else None
            param_alias = param_specifics['aliases'] if 'aliases' in param_specifics and param_name != 'input_model' else {}
            param_version = param_specifics['version'] if 'version' in param_specifics else None
            param_choices = param_specifics['choices'] if 'choices' in param_specifics else None

            # Bool params common setting
            if signature.parameters[param_name].annotation == bool and param_name != 'version':
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    action='store_true',
                    help=help_text,
                    default=signature.parameters[param_name].default)
            # File paths common setting
            elif param_name in filepath_args:
                action = action if action is not None else CanonicalizePathCheckExistenceAction
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    type=str if param_type is None else param_type,
                    nargs='*' if param_name == 'input_model' else '?',
                    action=action,
                    help=help_text,
                    default=None if param_name == 'input_model' else signature.parameters[param_name].default)
            # Other params
            else:
                additional_params = {}
                if param_version is not None:
                    additional_params['version'] = param_version
                if param_type is not None:
                    additional_params['type'] = param_type
                if param_choices is not None:
                    additional_params['choices'] = param_choices
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    help=help_text,
                    default=signature.parameters[param_name].default,
                    action=action,
                    **additional_params
                )


def get_common_cli_parser(parser: argparse.ArgumentParser = None):
    if not parser:
        parser = argparse.ArgumentParser()
    common_group = parser.add_argument_group('Framework-agnostic parameters')
    mo_convert_params = get_mo_convert_params()
    mo_convert_params_common = mo_convert_params['Framework-agnostic parameters:']

    from openvino.tools.ovc.version import VersionChecker

    # Command line tool specific params
    common_group.add_argument('--output_model',
                              help='This parameter is used to name output .xml/.bin files with converted model.')
    common_group.add_argument('--compress_to_fp16', type=check_bool, default=True,
                              help='Compress weights in output IR .xml/bin files to FP16.')
    common_group.add_argument('--version', action='version',
                              help='Print ovc version and exit.',
                              version='OpenVINO Model Converter (ovc) {}'.format(VersionChecker().get_ie_version()))
    add_args_by_description(common_group, mo_convert_params_common)
    return parser


def get_common_cli_options(model_name):
    d = OrderedDict()
    d['input_model'] = '- Path to the Input Model'
    d['output_dir'] = ['- Path for generated IR', lambda x: x if x != '.' else os.getcwd()]    # TODO: Consider removing
    d['output_model'] = ['- IR output name', lambda x: x if x else model_name]
    d['log_level'] = '- Log level'
    d['input'] = ['- Input layers', lambda x: x if x else 'Not specified, inherited from the model']
    d['output'] = ['- Output layers', lambda x: x if x else 'Not specified, inherited from the model']
    return d


def get_params_with_paths_list():
    return ['input_model', 'output_model', 'extensions']


def get_tf_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Conversion for TF

    Returns
    -------
        ArgumentParser instance
    """
    mo_convert_params_tf = get_mo_convert_params()['TensorFlow*-specific parameters:']

    tf_group = parser.add_argument_group('TensorFlow*-specific parameters')
    add_args_by_description(tf_group, mo_convert_params_tf)
    return parser


def get_onnx_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Conversion for ONNX

    Returns
    -------
        ArgumentParser instance
    """
    return parser


def get_all_cli_parser():
    """
    Specifies cli arguments for Model Conversion

    Returns
    -------
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser()

    get_common_cli_parser(parser=parser)
    get_tf_cli_parser(parser=parser)
    get_onnx_cli_parser(parser=parser)

    return parser


def remove_data_type_from_input_value(input_value: str):
    """
    Removes the type specification from the input string. The type specification is a string enclosed with curly braces.
    :param input_value: string passed as input to the "input" command line parameter
    :return: string without type specification
    """
    return re.sub(r'\{.*\}', '', input_value)


def get_data_type_from_input_value(input_value: str):
    """
    Returns the numpy data type corresponding to the data type specified in the input value string
    :param input_value: string passed as input to the "input" command line parameter
    :return: the corresponding numpy data type and None if the data type is not specified in the input value
    """
    data_type_match = re.match(r'.*\{(.*)\}.*', input_value)
    return destination_type_to_np_data_type(data_type_match.group(1)) if data_type_match is not None else None


def remove_shape_from_input_value(input_value: str):
    """
    Removes the shape specification from the input string. The shape specification is a string enclosed with square
    brackets.
    :param input_value: string passed as input to the "input" command line parameter
    :return: string without shape specification
    """
    assert '->' not in input_value, 'The function should not be called for input_value with constant value specified'
    return re.sub(r'[(\[]([0-9\.?,  -]*)[)\]]', '', input_value)


def get_shape_from_input_value(input_value: str):
    """
    Returns PartialShape corresponding to the shape specified in the input value string
    :param input_value: string passed as input to the "input" command line parameter
    :return: the corresponding shape and None if the shape is not specified in the input value
    """
    # remove the tensor value from the input_value first
    input_value = input_value.split('->')[0]

    # parse shape
    shape = re.findall(r'[(\[]([0-9\.\?,  -]*)[)\]]', input_value)
    if len(shape) == 0:
        shape = None
    elif len(shape) == 1 and shape[0] in ['', ' ']:
        # this shape corresponds to scalar
        shape = PartialShape([])
    elif len(shape) == 1:
        dims = re.split(r', *| +', shape[0])
        dims = list(filter(None, dims))
        shape = PartialShape([Dimension(dim) for dim in dims])
    else:
        raise Error("Wrong syntax to specify shape. Use \"input\" "
                    "\"node_name[shape]->value\"")
    return shape


def get_node_name_with_port_from_input_value(input_value: str):
    """
    Returns the node name (optionally with input/output port) from the input value
    :param input_value: string passed as input to the "input" command line parameter
    :return: the corresponding node name with input/output port
    """
    return remove_shape_from_input_value(remove_data_type_from_input_value(input_value.split('->')[0]))


def get_value_from_input_value(input_value: str):
    """
    Returns the value from the input value string
    :param input_value: string passed as input to the "input" command line parameter
    :return: the corresponding value or None if it is not specified
    """
    parts = input_value.split('->')
    value = None
    if len(parts) == 2:
        value = parts[1]
        if value[0] == '[' and value[-1] != ']' or value[0] != '[' and value[-1] == ']':
            raise Error("Wrong syntax to specify value. Use \"input\"=\"node_name[shape]->value\"")
        if '[' in value.strip(' '):
            value = value.replace('[', '').replace(']', '')
            if ',' in value:
                value = value.replace(' ', '')
                value = value.split(',')
            else:
                value = value.split(' ')
        if not isinstance(value, list):
            value = ast.literal_eval(value)
    elif len(parts) > 2:
        raise Error("Wrong syntax to specify value. Use \"input\"=\"node_name[shape]->value\"")
    return value


def partial_shape_prod(shape: [PartialShape, tuple]):
    assert not (isinstance(shape, PartialShape) and shape.is_dynamic), \
        "Unable to calculate prod for dynamic shape {}.".format(shape)

    prod = 1
    for dim in shape:
        prod *= dim.get_min_length()
    return prod


def parse_input_value(input_value: str):
    """
    Parses a value of the "input" command line parameter and gets a node name, shape and value.
    The node name includes a port if it is specified.
    Shape and value is equal to None if they are not specified.
    Parameters
    ----------
    input_value
        string with a specified node name, shape, value and data_type.
        E.g. 'node_name:0[4]{fp32}->[1.0 2.0 3.0 4.0]'

    Returns
    -------
        Node name, shape, value, data type
        E.g. 'node_name:0', '4', [1.0 2.0 3.0 4.0], np.float32
    """
    data_type = get_data_type_from_input_value(input_value)
    node_name = get_node_name_with_port_from_input_value(input_value)
    value = get_value_from_input_value(input_value)
    shape = get_shape_from_input_value(input_value)
    value_size = np.prod(len(value)) if isinstance(value, list) else 1

    if value is not None and shape is not None:
        for dim in shape:
            if isinstance(dim, Dimension) and dim.is_dynamic:
                raise Error("Cannot freeze input with dynamic shape: {}".format(shape))

    if shape is not None and value is not None and partial_shape_prod(shape) != value_size:
        raise Error("The shape '{}' of the input node '{}' does not correspond to the number of elements '{}' in the "
                    "value: {}".format(shape, node_name, value_size, value))
    return node_name if node_name else None, shape, value, data_type


def split_str_avoiding_square_brackets(s: str) -> list:
    """
    Splits a string by comma, but skips commas inside square brackets.
    :param s: string to split
    :return: list of strings split by comma
    """
    res = list()
    skipping = 0
    last_idx = 0
    for i, c in enumerate(s):
        if c == '[':
            skipping += 1
        elif c == ']':
            skipping -= 1
        elif c == ',' and skipping == 0:
            res.append(s[last_idx:i])
            last_idx = i + 1
    res.append(s[last_idx:])
    return res


def split_layouts_by_arrow(s: str) -> tuple:
    """
    Splits a layout string by first arrow (->).
    :param s: string to split
    :return: tuple containing source and target layouts
    """
    arrow = s.find('->')
    if arrow != -1:
        source_layout = s[:arrow]
        target_layout = s[arrow + 2:]
        if source_layout == '':
            source_layout = None
        if target_layout == '':
            target_layout = None
        return source_layout, target_layout
    else:
        return s, None


def validate_layout(layout: str):
    """
    Checks if layout is of valid format.
    :param layout: string containing layout
    :raises: if layout is incorrect
    """
    error_msg = 'Invalid layout parsed: {}'.format(layout)
    if layout:
        incorrect_brackets = xor(layout[0] == '[', layout[-1] == ']')
        if incorrect_brackets or layout[-1] == '-':
            error_msg += ', did you forget quotes?'
        else:
            valid_layout_re = re.compile(r'\[?[^\[\]\(\)\-\s]*\]?')
            if valid_layout_re.fullmatch(layout):
                return
        raise Error(error_msg)


def write_found_layout(name: str, found_layout: str, parsed: dict, dest: str = None):
    """
    Writes found layout data to the 'parsed' dict.
    :param name: name of the node to add layout
    :param found_layout: string containing layout for the node
    :param parsed: dict where result will be stored
    :param dest: type of the command line:
      * 'source' is "source_layout"
      * 'target' is "target_layout"
      * None is "layout"
    """
    s_layout = None
    t_layout = None
    if name in parsed:
        s_layout = parsed[name]['source_layout']
        t_layout = parsed[name]['target_layout']
    if dest == 'source':
        s_layout = found_layout
    elif dest == 'target':
        t_layout = found_layout
    else:
        s_layout, t_layout = split_layouts_by_arrow(found_layout)
    validate_layout(s_layout)
    validate_layout(t_layout)
    parsed[name] = {'source_layout': s_layout, 'target_layout': t_layout}


def write_found_layout_list(idx: int, found_layout: str, parsed: list, dest: str = None):
    """
    Writes found layout data to the 'parsed' dict.
    :param idx: idx of of the node to add layout
    :param found_layout: string containing layout for the node
    :param parsed: list where result will be stored
    :param dest: type of the command line:
      * 'source' is "source_layout"
      * 'target' is "target_layout"
      * None is "layout"
    """
    s_layout = None
    t_layout = None
    if idx < len(parsed):
        s_layout = parsed[idx]['source_layout']
        t_layout = parsed[idx]['target_layout']
    if dest == 'source':
        s_layout = found_layout
    elif dest == 'target':
        t_layout = found_layout
    else:
        s_layout, t_layout = split_layouts_by_arrow(found_layout)
    validate_layout(s_layout)
    validate_layout(t_layout)

    if idx < len(parsed):
        parsed[idx] = {'source_layout': s_layout, 'target_layout': t_layout}
    else:
        parsed.append({'source_layout': s_layout, 'target_layout': t_layout})


def parse_layouts_by_destination(s: str, parsed: dict, parsed_list: list, dest: str = None) -> None:
    """
    Parses layout command line to get all names and layouts from it. Adds all found data in the 'parsed' dict.
    :param s: string to parse
    :param parsed: dict where result will be stored
    :param dest: type of the command line:
      * 'source' is "source_layout"
      * 'target' is "target_layout"
      * None is "layout"
    """
    list_s = split_str_avoiding_square_brackets(s)
    if len(list_s) == 1 and (list_s[0][-1] not in ')]' or (list_s[0][0] == '[' and list_s[0][-1] == ']')):
        # single layout case
        write_found_layout('', list_s[0], parsed, dest)
    else:
        for idx, layout_str in enumerate(list_s):
            # case for: "name1(nhwc->[n,c,h,w])"
            p1 = re.compile(r'([^\[\]\(\)]*)\((\S+)\)')
            m1 = p1.match(layout_str)
            # case for: "name1[n,h,w,c]->[n,c,h,w]"
            p2 = re.compile(r'([^\[\]\(\)]*)(\[\S*\])')
            m2 = p2.match(layout_str)
            if m1:
                found_g = m1.groups()
            elif m2:
                found_g = m2.groups()
            else:
                # case for layout without name
                write_found_layout_list(idx, layout_str, parsed_list, dest)
                continue
            if len(found_g[0]) > 0:
                write_found_layout(found_g[0], found_g[1], parsed, dest)
            else:
                write_found_layout_list(idx, found_g[1], parsed_list, dest)


def get_layout_values(argv_layout: str = '', argv_source_layout: str = '', argv_target_layout: str = ''):
    """
    Parses layout string.
    :param argv_layout: string with a list of layouts passed as a "layout".
    :param argv_source_layout: string with a list of layouts passed as a "source_layout".
    :param argv_target_layout: string with a list of layouts passed as a "target_layout".
    :return: dict with names and layouts associated
    """
    if argv_layout and (argv_source_layout or argv_target_layout):
        raise Error("\"layout\" is used as well as \"source_layout\" and/or \"target_layout\" which is not allowed, please "
                    "use one of them.")
    res = {}
    res_list = []
    if argv_layout:
        parse_layouts_by_destination(argv_layout, res, res_list)
    if argv_source_layout:
        parse_layouts_by_destination(argv_source_layout, res, res_list, 'source')
    if argv_target_layout:
        parse_layouts_by_destination(argv_target_layout, res, res_list, 'target')
    if len(res) > 0 and len(res_list) > 0:
        raise Error("Some layout values are provided with names, and some without names. "
                    "Please provide ether all layouts with names or all layouts without names.")
    if len(res) > 0:
        return res
    else:
        return res_list


#TODO: Should be removed?
def parse_freeze_placeholder_values(argv_freeze_placeholder_with_value: str):
    """
    Parses parse_freeze_placeholder_values string.
    :param argv_freeze_placeholder_with_value: string information on freezing placeholders
    :return: dictionary where key is node name, value is node value.
    """
    placeholder_values = {}
    if argv_freeze_placeholder_with_value is not None:
        for plh_with_value in argv_freeze_placeholder_with_value.split(','):
            plh_with_value = plh_with_value.split('->')
            if len(plh_with_value) != 2:
                raise Error("Wrong replacement syntax. Use \"freeze_placeholder_with_value\" "
                            "\"node1_name->value1,node2_name->value2\"")
            node_name = plh_with_value[0]
            value = plh_with_value[1]
            if node_name in placeholder_values and placeholder_values[node_name] != value:
                raise Error("Overriding replacement value of the placeholder with name '{}': old value = {}, new value = {}"
                            ".".format(node_name, placeholder_values[node_name], value))
            if '[' in value.strip(' '):
                value = value.replace('[', '').replace(']', '').split(' ')
            placeholder_values[node_name] = value
    return placeholder_values


def get_freeze_placeholder_values(argv_input: str):
    """
    Parses values for placeholder freezing and input node names

    Parameters
    ----------
    argv_input
        string with a list of input layers: either an empty string, or strings separated with comma.
        'node_name1[shape1]->value1,node_name2[shape2]->value2,...'

    Returns
    -------
        parsed placeholders with values for freezing
        input nodes cleaned from shape info
    """
    placeholder_values = {}
    input_node_names = None

    if argv_input is not None:
        input_node_names = ''
        # walkthrough all input values and save values for freezing
        for input_value in split_inputs(argv_input):
            node_name, _, value, _ = parse_input_value(input_value)
            input_node_names = input_node_names + ',' + node_name  if input_node_names != '' else node_name
            if value is None: # no value is specified for freezing
                continue
            if node_name in placeholder_values and placeholder_values[node_name] != value:
                raise Error("Overriding replacement value of the placeholder with name '{}': old value = {}, new value = {}"
                            ".".format(node_name, placeholder_values[node_name], value))
            placeholder_values[node_name] = value

    return placeholder_values, input_node_names


def split_inputs(input_str):
    brakets_count = 0
    inputs = []
    while input_str:
        idx = 0
        for c in input_str:
            if c == '[':
                brakets_count += 1
            if c == ']':
                brakets_count -= 1
            if c == ',':
                if brakets_count != 0:
                    idx += 1
                    continue
                else:
                    break
            idx += 1
        if idx >= len(input_str)-1:
            inputs.append(input_str)
            break
        inputs.append(input_str[:idx])
        input_str = input_str[idx+1:]
    return inputs


def split_node_in_port(node_id: str):
    """Split node_id in form port:node to separate node and port, where port is converted to int"""
    if isinstance(node_id, str):
        separator = ':'
        parts = node_id.split(separator)
        if len(parts) > 1:
            if parts[0].isdigit():
                node_name = separator.join(parts[1:])
                try:
                    port = int(parts[0])
                    return node_name, port
                except ValueError as err:
                    log.warning('Didn\'t recognize port:node format for "{}" because port is not an integer.'.format(
                    node_id))
            else:
                node_name = separator.join(parts[:-1])
                try:
                    port = int(parts[-1])
                    return node_name, port
                except ValueError as err:
                    log.warning('Didn\'t recognize node:port format for "{}" because port is not an integer.'.format(
                    node_id))

    return node_id, None


def get_model_name(path_input_model: str) -> str:
    """
    Deduces model name by a given path to the input model
    Args:
        path_input_model: path to the input model

    Returns:
        name of the output IR
    """
    parsed_name, extension = os.path.splitext(os.path.basename(path_input_model))
    return 'model' if parsed_name.startswith('.') or len(parsed_name) == 0 else parsed_name


def get_model_name_from_args(argv: argparse.Namespace):
    if hasattr(argv, 'output_model') and argv.output_model:
        model_name = argv.output_model
    else:
        model_name = argv.input_model
        if isinstance(model_name, (tuple, list)) and len(model_name) > 0:
            model_name = model_name[0]
    return model_name


def get_absolute_path(path_to_file: str) -> str:
    """
    Deduces absolute path of the file by a given path to the file
    Args:
        path_to_file: path to the file

    Returns:
        absolute path of the file
    """
    file_path = os.path.expanduser(path_to_file)
    if not os.path.isabs(file_path):
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def isbool(value):
    try:
        strtobool(value)
        return True
    except ValueError:
        return False


def isdict(value):
    try:
        evaluated = ast.literal_eval(value)
        return isinstance(evaluated, dict)
    except ValueError:
        return False


def convert_string_to_real_type(value: str):
    if isdict(value):
        return ast.literal_eval(value)

    values = value.split(',')
    for i in range(len(values)):
        value = values[i]
        if value.isdigit():
            values[i] = int(value)
        elif isfloat(value):
            values[i] = float(value)
        elif isbool(value):
            values[i] = strtobool(value)

    return values[0] if len(values) == 1 else values


def check_positive(value):
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer value")

    return int_value


def check_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        if value.lower() not in ['true', 'false']:
            raise argparse.ArgumentTypeError("expected a True/False value")
        return value.lower() == 'true'
    else:
        raise argparse.ArgumentTypeError("expected a bool or str type")


def depersonalize(value: str, key: str):
    dir_keys = [
        'output_dir', 'extensions', 'saved_model_dir', 'tensorboard_logdir', 'caffe_parser_path'
    ]
    if isinstance(value, list):
        updated_value = []
        for elem in value:
            updated_value.append(depersonalize(elem, key))
        return updated_value

    if not isinstance(value, str):
        return value
    res = []
    for path in value.split(','):
        if os.path.isdir(path) and key in dir_keys:
            res.append('DIR')
        elif os.path.isfile(path):
            res.append(os.path.join('DIR', os.path.split(path)[1]))
        else:
            res.append(path)
    return ','.join(res)

def get_available_front_ends(fem=None):
    # Use this function as workaround to avoid IR frontend usage by MO
    if fem is None:
        return []
    available_moc_front_ends = fem.get_available_front_ends()
    if 'ir' in available_moc_front_ends:
        available_moc_front_ends.remove('ir')

    return available_moc_front_ends
