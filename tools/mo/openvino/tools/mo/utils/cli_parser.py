# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import logging as log
import os
import re
from collections import OrderedDict, namedtuple
from itertools import zip_longest
from pathlib import Path
from operator import xor
from typing import List, Union
import numbers
import inspect

import numpy as np
from openvino.runtime import Layout, PartialShape, Dimension, Shape, Type

import openvino
from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg, get_mo_root_dir
from openvino.tools.mo.utils.help import get_convert_model_help_specifics, get_to_string_methods_for_params


def strtobool(value):
    """
    Converts a string representation to true or false.

    :param value: a string which will be converted to bool.
    :return: boolean value of the input string.
    """
    value = value.lower()
    if value in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif value in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError(f"Invalid truth value: {value}.")


def extension_path_to_str_or_extensions_class(extension):
    if isinstance(extension, str):
        return extension
    elif isinstance(extension, Path):
        return str(extension)
    else:
        # Return unknown object as is.
        # The type of the object will be checked by frontend.add_extension() method
        return extension


def transformations_config_to_str(value):
    if value is None:
        return value
    return extension_path_to_str_or_extensions_class(value)


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
        return openvino.tools.mo.InputCutInfo(node_name,
                                              PartialShape(shape) if shape is not None else None,
                                              data_type,
                                              value)
    if isinstance(input, openvino.tools.mo.InputCutInfo):
        # Wrap input.shape to PartialShape if possible and wrap to InputCutInfo
        return openvino.tools.mo.InputCutInfo(input.name,
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
        return openvino.tools.mo.InputCutInfo(name,
                                              PartialShape(shape) if shape is not None else None,
                                              inp_type,
                                              None)
    # Case when only type is set
    if isinstance(input, (type, Type)):
        return openvino.tools.mo.InputCutInfo(None, None, input, None)

    # We don't expect here single unnamed value. If list of int is set it is considered as shape.
    # Setting of value is expected only using InputCutInfo or string analog.

    raise Exception("Unexpected object provided for input. Expected openvino.toos.mo.InputCutInfo "
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
            inputs.append(openvino.tools.mo.InputCutInfo(node_name,
                                                         PartialShape(shape) if shape is not None else None,
                                                         data_type,
                                                         value))
        return inputs
    if isinstance(input, openvino.tools.mo.InputCutInfo):
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


def input_shape_to_input_cut_info(input_shape: [str, Shape, PartialShape, list, tuple], inputs: list):
    """
    Parses 'input_shape' to list of PartialShape and updates 'inputs'.
    :param input_shape: input shapes passed by user
    :param inputs: list of InputCutInfo with information from 'input' parameter
    """
    if input_shape is None:
        return
    if isinstance(input_shape, str):
        # Split input_shape to list of string
        input_shape = split_shapes(input_shape)
    if isinstance(input_shape, (Shape, PartialShape)):
        # Whap single shape to list
        input_shape = [input_shape]
    if isinstance(input_shape, (list, tuple)):
        # Check case when single shape is passed as list or tuple
        if len(input_shape) > 0 and isinstance(input_shape[0], (int, Dimension)):
            input_shape = [input_shape]

        if len(inputs) > 0 and len(input_shape) > 0:
            assert len(inputs) == len(input_shape), "Different numbers of inputs were specified in \"input\" parameter " \
                    "and \"input_shapes\". \"input\" has {} items, \"input_shape\" has {} item.".format(len(inputs), len(input_shape))

        # Update inputs with information from 'input_shape'
        if len(inputs) > 0:
            for idx, shape in enumerate(input_shape):
                shape = PartialShape(shape)
                assert inputs[idx].shape is None, "Shape was set in both \"input\" and in \"input_shape\" parameter." \
                                                  "Please use either \"input\" or \"input_shape\" for shape setting."
                inputs[idx] = openvino.tools.mo.InputCutInfo(inputs[idx].name, shape, inputs[idx].type, inputs[idx].value)

        else:
            for shape in input_shape:
                inputs.append(openvino.tools.mo.InputCutInfo(None, PartialShape(shape), None, None))
        return

    raise Exception("Unexpected object provided for input_shape. Expected PartialShape, Shape, tuple, list or str. "
                    "Got {}".format(type(input_shape)))


def freeze_placeholder_to_input_cut_info(argv_freeze_placeholder_with_value: str, inputs: list):
    """
    Parses 'argv_freeze_placeholder_with_value' to dictionary and collects unnamed inputs from 'inputs' to list.
    :param argv_freeze_placeholder_with_value: string set by user.
    As it was planned to be deprecated no Python analogs were made.
    :param inputs: list of InputCutInfo with information from 'input' parameter
    :returns (placeholder_values, unnamed_placeholder_values), where
    placeholder_values - dictionary where key is node name, value is node value,
    unnamed_placeholder_values - list with unnamed node values
    """
    # Parse argv_freeze_placeholder_with_value to dictionary with names and values
    placeholder_values = parse_freeze_placeholder_values(argv_freeze_placeholder_with_value)
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


def mean_scale_value_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        values_str = []
        for op_name, val in value.items():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + value_to_str(val, ","))
        return ",".join(values_str)
    if isinstance(value, list) or isinstance(value, tuple):
        list_of_lists = False
        for val in value:
            if isinstance(val, list) or isinstance(val, tuple):
                list_of_lists = True
                break
        if list_of_lists:
            values_str = []
            for val in value:
                values_str.append(value_to_str(val, ","))
            return ",".join(values_str)
        else:
            return value_to_str(value, ",")
    return value_to_str(value, ",")


def layout_to_str(layout):
    if isinstance(layout, str):
        return layout
    if isinstance(layout, Layout):
        return layout.to_string()
    raise Exception("Incorrect layout type. Expected Layout or string or dictionary, "
                    "where key is operation name and value is layout or list of layouts, got {}".format(type(layout)))


def source_target_layout_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        values_str = []
        for op_name, layout in value.items():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + "(" + layout_to_str(layout) + ")")
        return ",".join(values_str)

    return layout_to_str(value)


def layoutmap_to_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, openvino.tools.mo.LayoutMap):
        assert value.source_layout is not None, "Incorrect layout map. 'source_layout' should be set."
        source_layout = layout_to_str(value.source_layout)
        if value.target_layout is not None:
            target_layout = layout_to_str(value.target_layout)
            source_layout += "->" + target_layout
        return source_layout
    return layout_to_str(value)


def layout_param_to_str(value):
    # default empty value
    if isinstance(value, tuple) and len(value) == 0:
        return value

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        values_str = []
        for op_name, layout in value.items():
            if not isinstance(op_name, str):
                raise Exception("Incorrect operation name type. Expected string, got {}".format(type(op_name)))
            values_str.append(op_name + "(" + layoutmap_to_str(layout) + ")")
        return ",".join(values_str)
    if isinstance(value, openvino.tools.mo.LayoutMap):
        return layoutmap_to_str(value)
    if isinstance(value, list) or isinstance(value, tuple):
        values_str = []
        for layout in value:
            values_str.append(layoutmap_to_str(layout))
        return ",".join(values_str)

    return layoutmap_to_str(value)


def batch_to_int(value):
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, Dimension):
        if not value.is_static:
            # TODO: Ticket 88676
            raise Exception("Dynamic batch for \"batch\" parameter is not supported.")
        else:
            return value.get_length()
    raise Exception("Incorrect batch value. Expected int, got {}.".format(type(value)))


def transform_param_value_to_str(value):
    # This function supports parsing of parameters of MakeStateful, LowLatency2, Pruning.
    # If available transforms list is extended this method should be extended for new transforms.
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, dict):
        # param_res_names dictionary for MakeStateful transform
        values_str = []
        for input_name, output_name in value.items():
            assert isinstance(input_name, str), "Incorrect input name. " \
                                                "Expected string, got {}".format(type(input_name))
            assert isinstance(output_name, str), "Incorrect output name. " \
                                                 "Expected string, got {}".format(type(output_name))
            values_str.append("\'{}\':\'{}\'".format(input_name, output_name))
        return "{" + ','.join(values_str) + "}"
    raise Exception("Unknown parameter type.")


def transform_to_str(value):
    from openvino.tools.mo.back.offline_transformations import get_available_transformations   # pylint: disable=no-name-in-module,import-error

    if isinstance(value, str):
        return value

    if isinstance(value, tuple):
        assert 1 <= len(value) <= 2, "Incorrect definition of transformation in transform argument: " \
                                     "expected two elements in tuple, provided {}. " \
                                     "Supported transforms are: {}".format(
            len(value),
            list(get_available_transformations().keys()))
        transform_name = value[0]
        assert isinstance(transform_name, str), "Incorrect transform name type. " \
                                                "Expected string, got {}".format(type(transform_name))
        if len(value) == 2:
            params = value[1]
            assert isinstance(params, dict), "Incorrect transform params type. " \
                                             "Expected dictionary, got {}".format(type(params))
            params_str_list = []
            for param_name, val in params.items():
                assert isinstance(param_name, str), "Incorrect transform parameter name type. " \
                                                    "Expected string, got {}".format(type(param_name))
                val_str = transform_param_value_to_str(val)
                params_str_list.append(param_name + "=" + val_str)
            transform_name += '[' + ','.join(params_str_list) + ']'
        return transform_name
    raise Exception("Incorrect transform type. Expected tuple with transform name and "
                    "dictionary with transform parameters. Got object of type {}".format(type(value)))


def transform_param_to_str(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        transforms_str = []
        for transform in value:
            transforms_str.append(transform_to_str(transform))
        return ','.join(transforms_str)
    return transform_to_str(value)


ParamDescription = namedtuple("ParamData",
                              ["description", "cli_tool_description", "to_string"])


def get_mo_convert_params():
    mo_convert_docs = openvino.tools.mo.convert_model.__doc__
    mo_convert_params = {}
    group = "Optional parameters:"
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

    # TODO: remove this when internal converting of params to string is removed
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
        if 'fusing' in option_string:
            dep_msg += 'Please use --finegrain_fusing cli option instead'
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


def canonicalize_and_check_paths(values: Union[str, List[str]], param_name,
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
                path_from_mo_root = get_mo_root_dir() + '/mo/' + val
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
        setattr(namespace, self.dest, ','.join(list_of_paths))


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
        setattr(namespace, self.dest, ','.join(list_of_paths))


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
    signature = inspect.signature(openvino.tools.mo.convert_model)
    filepath_args = get_params_with_paths_list()
    cli_tool_specific_descriptions = get_convert_model_help_specifics()
    for param_name, param_description in params_description.items():
        if param_name == 'help':
            continue
        cli_param_name = "--"+param_name
        if cli_param_name not in args_group._option_string_actions:
            # Get parameter specifics
            param_specifics = cli_tool_specific_descriptions[param_name] if param_name in \
                                                                            cli_tool_specific_descriptions else {}
            help_text = param_specifics['description'] if 'description' in param_specifics \
                else param_description.description
            action = param_specifics['action'] if 'action' in param_specifics else None
            param_type = param_specifics['type'] if 'type' in param_specifics else None
            param_alias = param_specifics['aliases'] if 'aliases' in param_specifics else {}
            param_version = param_specifics['version'] if 'version' in param_specifics else None
            param_choices = param_specifics['choices'] if 'choices' in param_specifics else None

            # Bool params common setting
            if signature.parameters[param_name].annotation == bool and param_name != 'version':
                default_flag = signature.parameters[param_name].default
                # tools.mo.convert_model by default does not compress,
                # but if we convert from cli we need to compress_to_fp16 if user did not specify otherwise
                if param_name == 'compress_to_fp16':
                    default_flag = True
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    type=check_bool if param_type is None else param_type,
                    nargs="?",
                    const=True,
                    help=help_text,
                    default=default_flag)
            # File paths common setting
            elif param_name in filepath_args:
                action = action if action is not None else CanonicalizePathCheckExistenceAction
                args_group.add_argument(
                    cli_param_name, *param_alias,
                    type=str if param_type is None else param_type,
                    action=action,
                    help=help_text,
                    default=signature.parameters[param_name].default)
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

    # Command line tool specific params
    common_group.add_argument('--model_name', '-n',
                              help='Model_name parameter passed to the final create_ir transform. ' +
                                   'This parameter is used to name ' +
                                   'a network in a generated IR and output .xml/.bin files.')
    common_group.add_argument('--output_dir', '-o',
                              help='Directory that stores the generated IR. ' +
                                   'By default, it is the directory from where the Model Conversion is launched.',
                              default=get_absolute_path('.'),
                              action=CanonicalizePathAction,
                              type=writable_dir)

    # Deprecated params
    common_group.add_argument('--freeze_placeholder_with_value',
                              help='Replaces input layer with constant node with '
                                   'provided value, for example: "node_name->True". '
                                   'It will be DEPRECATED in future releases. '
                                   'Use "input" option to specify a value for freezing.',
                              default=None)
    common_group.add_argument('--static_shape',
                              help='Enables IR generation for fixed input shape (folding `ShapeOf` operations and '
                                   'shape-calculating sub-graphs to `Constant`). Changing model input shape using '
                                   'the OpenVINO Runtime API in runtime may fail for such an IR.',
                              action='store_true', default=False)
    common_group.add_argument("--use_new_frontend",
                              help='Force the usage of new Frontend for model conversion into IR. '
                                   'The new Frontend is C++ based and is available for ONNX* and PaddlePaddle* models. '
                                   'Model Conversion API uses new Frontend for ONNX* and PaddlePaddle* by default that means '
                                   '`use_new_frontend` and `use_legacy_frontend` options are not specified.',
                              action='store_true', default=False)
    common_group.add_argument("--use_legacy_frontend",
                              help='Force the usage of legacy Frontend for model conversion into IR. '
                                   'The legacy Frontend is Python based and is available for TensorFlow*, ONNX*, '
                                   'Caffe*, and Kaldi* models.',
                              action='store_true', default=False)
    add_args_by_description(common_group, mo_convert_params_common)
    return parser


def get_common_cli_options(model_name):
    d = OrderedDict()
    d['input_model'] = '- Path to the Input Model'
    d['output_dir'] = ['- Path for generated IR', lambda x: x if x != '.' else os.getcwd()]
    d['model_name'] = ['- IR output name', lambda x: x if x else model_name]
    d['log_level'] = '- Log level'
    d['batch'] = ['- Batch', lambda x: x if x else 'Not specified, inherited from the model']
    d['input'] = ['- Input layers', lambda x: x if x else 'Not specified, inherited from the model']
    d['output'] = ['- Output layers', lambda x: x if x else 'Not specified, inherited from the model']
    d['input_shape'] = ['- Input shapes', lambda x: x if x else 'Not specified, inherited from the model']
    d['source_layout'] = ['- Source layout', lambda x: x if x else 'Not specified']
    d['target_layout'] = ['- Target layout', lambda x: x if x else 'Not specified']
    d['layout'] = ['- Layout', lambda x: x if x else 'Not specified']
    d['mean_values'] = ['- Mean values', lambda x: x if x else 'Not specified']
    d['scale_values'] = ['- Scale values', lambda x: x if x else 'Not specified']
    d['scale'] = ['- Scale factor', lambda x: x if x else 'Not specified']
    d['transform'] = ['- User transformations', lambda x: x if x else 'Not specified']
    d['reverse_input_channels'] = '- Reverse input channels'
    d['static_shape'] = '- Enable IR generation for fixed input shape'
    d['transformations_config'] = '- Use the transformations config file'
    return d


def get_advanced_cli_options():
    d = OrderedDict()
    d['use_legacy_frontend'] = '- Force the usage of legacy Frontend for model conversion into IR'
    d['use_new_frontend'] = '- Force the usage of new Frontend for model conversion into IR'
    return d


def get_caffe_cli_options():
    d = {
        'input_proto': ['- Path to the Input prototxt', lambda x: x],
        'caffe_parser_path': ['- Path to Python Caffe* parser generated from caffe.proto', lambda x: x],
        'k': '- Path to CustomLayersMapping.xml',
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_tf_cli_options():
    d = {
        'input_model_is_text': '- Input model in text protobuf format',
        'tensorflow_custom_operations_config_update': '- Update the configuration file with input/output node names',
        'tensorflow_object_detection_api_pipeline_config': '- Use configuration file used to generate the model with '
                                                           'Object Detection API',
        'tensorflow_custom_layer_libraries': '- List of shared libraries with TensorFlow custom layers implementation',
        'tensorboard_logdir': '- Path to model dump for TensorBoard'
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_kaldi_cli_options():
    d = {
        'counts': '- A file name with full path to the counts file or empty string if you want to use counts from model',
        'remove_output_softmax': '- Removes the SoftMax layer that is the output layer',
        'remove_memory': '- Removes the Memory layer and use additional inputs and outputs instead'
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_onnx_cli_options():
    d = {
    }

    return OrderedDict(sorted(d.items(), key=lambda t: t[0]))


def get_params_with_paths_list():
    return ['input_model', 'output_dir', 'caffe_parser_path', 'extensions', 'k', 'output_dir',
            'input_checkpoint', 'input_meta_graph', 'input_proto', 'input_symbol',
            'pretrained_model_name', 'saved_model_dir', 'tensorboard_logdir',
            'tensorflow_custom_layer_libraries', 'tensorflow_custom_operations_config_update',
            'tensorflow_object_detection_api_pipeline_config',
            'transformations_config']


def get_caffe_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Conversion for Caffe*

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)

    caffe_group = parser.add_argument_group('Caffe*-specific parameters')
    mo_convert_params_caffe = get_mo_convert_params()['Caffe*-specific parameters:']
    add_args_by_description(caffe_group, mo_convert_params_caffe)
    return parser


def get_tf_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Conversion for TF

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)
    mo_convert_params_tf = get_mo_convert_params()['TensorFlow*-specific parameters:']

    tf_group = parser.add_argument_group('TensorFlow*-specific parameters')
    add_args_by_description(tf_group, mo_convert_params_tf)
    return parser


def get_kaldi_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Conversion for Kaldi*

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)

    kaldi_group = parser.add_argument_group('Kaldi-specific parameters')
    mo_convert_params_kaldi = get_mo_convert_params()['Kaldi-specific parameters:']
    add_args_by_description(kaldi_group, mo_convert_params_kaldi)
    return parser


def get_onnx_cli_parser(parser: argparse.ArgumentParser = None):
    """
    Specifies cli arguments for Model Conversion for ONNX

    Returns
    -------
        ArgumentParser instance
    """
    if not parser:
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')
        get_common_cli_parser(parser=parser)

    return parser


def get_all_cli_parser():
    """
    Specifies cli arguments for Model Conversion

    Returns
    -------
        ArgumentParser instance
    """
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')
    mo_convert_params_optional = get_mo_convert_params()['Optional parameters:']
    add_args_by_description(parser, mo_convert_params_optional)

    get_common_cli_parser(parser=parser)
    get_tf_cli_parser(parser=parser)
    get_caffe_cli_parser(parser=parser)
    get_kaldi_cli_parser(parser=parser)
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
    return node_name, shape, value, data_type


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


def get_freeze_placeholder_values(argv_input: str, argv_freeze_placeholder_with_value: str):
    """
    Parses values for placeholder freezing and input node names

    Parameters
    ----------
    argv_input
        string with a list of input layers: either an empty string, or strings separated with comma.
        'node_name1[shape1]->value1,node_name2[shape2]->value2,...'
    argv_freeze_placeholder_with_value
        string with a list of input shapes: either an empty string, or tuples separated with comma.
        'placeholder_name1->value1, placeholder_name2->value2,...'

    Returns
    -------
        parsed placeholders with values for freezing
        input nodes cleaned from shape info
    """
    placeholder_values = parse_freeze_placeholder_values(argv_freeze_placeholder_with_value)
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



def split_shapes(argv_input_shape: str):
    range_reg = r'([0-9]*\.\.[0-9]*)'
    first_digit_reg = r'([0-9 ]+|-1|\?|{})'.format(range_reg)
    next_digits_reg = r'(,{})*'.format(first_digit_reg)
    tuple_reg = r'((\({}{}\))|(\[{}{}\]))'.format(first_digit_reg, next_digits_reg,
                                                  first_digit_reg, next_digits_reg)

    full_reg = r'^{}(\s*,\s*{})*$|^$'.format(tuple_reg, tuple_reg)
    if not re.match(full_reg, argv_input_shape):
        raise Error('Input shape "{}" cannot be parsed. ' + refer_to_faq_msg(57), argv_input_shape)
    return re.findall(r'[(\[]([0-9,\.\? -]+)[)\]]', argv_input_shape)

def get_placeholder_shapes(argv_input: str, argv_input_shape: str, argv_batch=None):
    """
    Parses input layers names and input shapes from the cli and returns the parsed object.
    All shapes are specified only through one command line option either "input" or "input_shape".

    Parameters
    ----------
    argv_input
        string with a list of input layers: either an empty string, or strings separated with comma.
        E.g. 'inp1,inp2', 'node_name1[shape1]->value1,node_name2[shape2]->value2'
    argv_input_shape
        string with a list of input shapes: either an empty string, or tuples separated with comma.
        E.g. '[1,2],[3,4]'.
        Only positive integers are accepted.
        '?' marks dynamic dimension.
        Partial shape is specified with ellipsis. E.g. '[1..10,2,3]'
    argv_batch
        integer that overrides batch size in input shape

    Returns
    -------
        parsed shapes in form of {'name of input':tuple} if names of inputs are provided with shapes
        parsed shapes in form of {'name of input':None} if names of inputs are provided without shapes
        tuple if only one shape is provided and no input name
        None if neither shape nor input were provided
    """
    if argv_input_shape and argv_batch:
        raise Error("Both \"input_shape\" and \"batch\" were provided. Please provide only one of them. " +
                    refer_to_faq_msg(56))

    # attempt to extract shapes from "input" parameters
    placeholder_shapes = dict()
    placeholder_data_types = dict()
    are_shapes_specified_through_input = False
    inputs_list = list()
    if argv_input:
        for input_value in split_inputs(argv_input):
            node_name, shape, _, data_type = parse_input_value(input_value)
            placeholder_shapes[node_name] = shape
            inputs_list.append(node_name)
            if data_type is not None:
                placeholder_data_types[node_name] = data_type
            if shape is not None:
                are_shapes_specified_through_input = True

    if argv_input_shape and are_shapes_specified_through_input:
        raise Error("Shapes are specified using both \"input\" and \"input_shape\" command-line parameters, but only one "
                    "parameter is allowed.")

    if argv_batch and are_shapes_specified_through_input:
        raise Error("Shapes are specified using both \"input\" and \"batch\" command-line parameters, but only one "
                    "parameter is allowed.")

    if are_shapes_specified_through_input:
        return inputs_list, placeholder_shapes, placeholder_data_types

    shapes = list()
    inputs = list()
    inputs_list = list()
    placeholder_shapes = None


    if argv_input_shape:
        shapes = split_shapes(argv_input_shape)

    if argv_input:
        inputs = split_inputs(argv_input)
    inputs = [remove_data_type_from_input_value(inp) for inp in inputs]

    # check number of shapes with no input provided
    if argv_input_shape and not argv_input:
        placeholder_shapes = [PartialShape(shape) for shape in shapes]
        if len(placeholder_shapes) == 1:
            placeholder_shapes = PartialShape(placeholder_shapes[0])
    # check if number of shapes does not match number of passed inputs
    elif argv_input and (len(shapes) == len(inputs) or len(shapes) == 0):
        # clean inputs from values for freezing
        inputs_without_value = list(map(lambda x: x.split('->')[0], inputs))
        placeholder_shapes = dict(zip_longest(inputs_without_value,
                                              map(lambda x: PartialShape(x) if x else None, shapes)))
        for inp in inputs:
            if '->' not in inp:
                inputs_list.append(inp)
                continue
            shape = placeholder_shapes[inp.split('->')[0]]
            inputs_list.append(inp.split('->')[0])

            if shape is None:
                continue
            for dim in shape:
                if isinstance(dim, Dimension) and not dim.is_static:
                    raise Error("Cannot freeze input with dynamic shape: {}".format(shape))

    elif argv_input:
        raise Error('Please provide each input layers with an input layer shape. ' + refer_to_faq_msg(58))

    return inputs_list, placeholder_shapes, placeholder_data_types


def parse_tuple_pairs(argv_values: str):
    """
    Gets mean/scale values from the given string parameter
    Parameters
    ----------
    argv_values
        string with a specified input name and  list of mean values: either an empty string, or a tuple
        in a form [] or ().
        E.g. 'data(1,2,3)' means 1 for the RED channel, 2 for the GREEN channel, 3 for the BLUE channel for the data
        input layer, or tuple of values in a form [] or () if input is specified separately, e.g. (1,2,3),[4,5,6].

    Returns
    -------
        dictionary with input name and tuple of values or list of values if mean/scale value is specified with input,
        e.g.:
        "data(10,20,30),info(11,22,33)" -> { 'data': [10,20,30], 'info': [11,22,33] }
        "(10,20,30),(11,22,33)" -> [mo_array(10,20,30), mo_array(11,22,33)]
    """
    res = {}
    if not argv_values:
        return res

    matches = [m for m in re.finditer(r'[(\[]([0-9., -]+)[)\]]', argv_values, re.IGNORECASE)]

    error_msg = 'Mean/scale values should consist of name and values specified in round or square brackets ' \
                'separated by comma, e.g. data(1,2,3),info[2,3,4],egg[255] or data(1,2,3). Or just plain set of ' \
                'values without names: (1,2,3),(2,3,4) or [1,2,3],[2,3,4].' + refer_to_faq_msg(101)
    if not matches:
        raise Error(error_msg, argv_values)

    name_start_idx = 0
    name_was_present = False
    for idx, match in enumerate(matches):
        input_name = argv_values[name_start_idx:match.start(0)]
        name_start_idx = match.end(0) + 1
        tuple_value = np.fromstring(match.groups()[0], dtype=float, sep=',')

        if idx != 0 and (name_was_present ^ bool(input_name)):
            # if node name firstly was specified and then subsequently not or vice versa
            # e.g. (255),input[127] or input(255),[127]
            raise Error(error_msg, argv_values)

        name_was_present = True if input_name != "" else False
        if name_was_present:
            res[input_name] = tuple_value
        else:
            res[idx] = tuple_value

    if not name_was_present:
        # return a list instead of a dictionary
        res = sorted(res.values(), key=lambda v: v[0])
    return res


def get_tuple_values(argv_values: str or tuple, num_exp_values: int = 3, t=float or int):
    """
    Gets mean values from the given string parameter
    Args:
        argv_values: string with list of mean values: either an empty string, or a tuple in a form [] or ().
        E.g. '(1,2,3)' means 1 for the RED channel, 2 for the GREEN channel, 4 for the BLUE channel.
        t: either float or int
        num_exp_values: number of values in tuple

    Returns:
        tuple of values
    """

    digit_reg = r'(-?[0-9. ]+)' if t == float else r'(-?[0-9 ]+)'

    assert num_exp_values > 1, 'Can not parse tuple of size 1'
    content = r'{0}\s*,{1}\s*{0}'.format(digit_reg, (digit_reg + ',') * (num_exp_values - 2))
    tuple_reg = r'((\({0}\))|(\[{0}\]))'.format(content)

    if isinstance(argv_values, tuple) and not len(argv_values):
        return argv_values

    if not len(argv_values) or not re.match(tuple_reg, argv_values):
        raise Error('Values "{}" cannot be parsed. ' +
                    refer_to_faq_msg(59), argv_values)

    mean_values_matches = re.findall(r'[(\[]([0-9., -]+)[)\]]', argv_values)

    for mean in mean_values_matches:
        if len(mean.split(',')) != num_exp_values:
            raise Error('{} channels are expected for given values. ' +
                        refer_to_faq_msg(60), num_exp_values)

    return mean_values_matches


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


def get_mean_scale_dictionary(mean_values, scale_values, argv_input: list):
    """
    This function takes mean_values and scale_values, checks and processes them into convenient structure

    Parameters
    ----------
    mean_values dictionary, contains input name and mean values passed py user (e.g. {data: np.array[102.4, 122.1, 113.9]}),
    or list containing values (e.g. np.array[102.4, 122.1, 113.9])
    scale_values dictionary, contains input name and scale values passed py user (e.g. {data: np.array[102.4, 122.1, 113.9]})
    or list containing values (e.g. np.array[102.4, 122.1, 113.9])

    Returns
    -------
    The function returns a dictionary e.g.
    mean = { 'data': np.array, 'info': np.array }, scale = { 'data': np.array, 'info': np.array }, input = "data, info" ->
     { 'data': { 'mean': np.array, 'scale': np.array }, 'info': { 'mean': np.array, 'scale': np.array } }

    """
    res = {}
    # collect input names
    if argv_input:
        inputs = [get_node_name_with_port_from_input_value(input_value) for input_value in split_inputs(argv_input)]
    else:
        inputs = []
        if type(mean_values) is dict:
            inputs = list(mean_values.keys())
        if type(scale_values) is dict:
            for name in scale_values.keys():
                if name not in inputs:
                    inputs.append(name)

    # create unified object containing both mean and scale for input
    if type(mean_values) is dict and type(scale_values) is dict:
        if not mean_values and not scale_values:
            return res

        for inp_scale in scale_values.keys():
            if inp_scale not in inputs:
                raise Error("Specified scale_values name '{}' do not match to any of inputs: {}. "
                            "Please set 'scale_values' that correspond to values from input.".format(inp_scale, inputs))

        for inp_mean in mean_values.keys():
            if inp_mean not in inputs:
                raise Error("Specified mean_values name '{}' do not match to any of inputs: {}. "
                            "Please set 'mean_values' that correspond to values from input.".format(inp_mean, inputs))

        for inp in inputs:
            inp, port = split_node_in_port(inp)
            if inp in mean_values or inp in scale_values:
                res.update(
                    {
                        inp: {
                            'mean':
                                mean_values[inp] if inp in mean_values else None,
                            'scale':
                                scale_values[inp] if inp in scale_values else None
                        }
                    }
                )
        return res

    # user specified input and mean/scale separately - we should return dictionary
    if inputs:
        if mean_values and scale_values:
            if len(inputs) != len(mean_values):
                raise Error('Numbers of inputs and mean values do not match. ' +
                            refer_to_faq_msg(61))
            if len(inputs) != len(scale_values):
                raise Error('Numbers of inputs and scale values do not match. ' +
                            refer_to_faq_msg(62))

            data = list(zip(mean_values, scale_values))

            for i in range(len(data)):
                res.update(
                    {
                        inputs[i]: {
                            'mean':
                                data[i][0],
                            'scale':
                                data[i][1],

                        }
                    }
                )
            return res
        # only mean value specified
        if mean_values:
            data = list(mean_values)
            for i in range(len(data)):
                res.update(
                    {
                        inputs[i]: {
                            'mean':
                                data[i],
                            'scale':
                                None

                        }
                    }
                )
            return res

        # only scale value specified
        if scale_values:
            data = list(scale_values)
            for i in range(len(data)):
                res.update(
                    {
                        inputs[i]: {
                            'mean':
                                None,
                            'scale':
                                data[i]

                        }
                    }
                )
            return res
    # mean and/or scale are specified without inputs
    return list(zip_longest(mean_values, scale_values))


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
    model_name = "<UNKNOWN_NAME>"
    if hasattr(argv, 'model_name'):
        if argv.model_name:
            model_name = argv.model_name
        elif argv.input_model:
            model_name = get_model_name(argv.input_model)
        elif argv.saved_model_dir:
            model_name = "saved_model"
        elif argv.input_meta_graph:
            model_name = get_model_name(argv.input_meta_graph)
        elif argv.input_symbol:
            model_name = get_model_name(argv.input_symbol)
        argv.model_name = model_name
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


def parse_transform(transform: str) -> list:
    transforms = []

    if len(transform) == 0:
        return transforms

    all_transforms = re.findall(r"([a-zA-Z0-9]+)(\[([^\]]+)\])*(,|$)", transform)

    # Check that all characters were matched otherwise transform key value is invalid
    key_len = len(transform)
    for transform in all_transforms:
        # In regexp we have 4 groups where 1st group - transformation_name,
        #                                  2nd group - [args],
        #                                  3rd group - args, <-- nested group
        #                                  4th group - EOL
        # And to check that regexp matched all string we decrease total length by the length of matched groups (1,2,4)
        # In case if no arguments were given to transformation then 2nd and 3rd groups will be empty.
        if len(transform) != 4:
            raise Error("Unexpected transform key structure: {}".format(transform))
        key_len -= len(transform[0]) + len(transform[1]) + len(transform[3])

    if key_len != 0:
        raise Error("Unexpected transform key structure: {}".format(transform))

    for transform in all_transforms:
        name = transform[0]
        args = transform[2]

        args_dict = {}

        if len(args) != 0:
            for arg in args.split(';'):
                m = re.match(r"^([_a-zA-Z]+)=(.+)$", arg)
                if not m:
                    raise Error("Unrecognized attributes for transform key: {}".format(transform))

                args_dict[m.group(1)] = convert_string_to_real_type(m.group(2))

        transforms.append((name, args_dict))

    return transforms


def check_available_transforms(transforms: list):
    """
    This function check that transformations specified by user are available.
    :param transforms: list of user specified transformations
    :return: raises an Error if transformation is not available
    """
    from openvino.tools.mo.back.offline_transformations import get_available_transformations   # pylint: disable=no-name-in-module,import-error
    available_transforms = get_available_transformations()

    missing_transformations = []
    for name, _ in transforms:
        if name not in available_transforms.keys():
            missing_transformations.append(name)

    if len(missing_transformations) != 0:
        raise Error('Following transformations ({}) are not available. '
                    'List with available transformations ({})'.format(','.join(missing_transformations),
                                                                      ','.join(available_transforms.keys())))
    return True


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
