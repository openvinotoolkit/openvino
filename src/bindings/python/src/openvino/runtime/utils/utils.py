# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import functools
import os
import re
import warnings
from typing import Callable
from typing import Iterable, Union

import numpy as np
from openvino.runtime.utils.error import Error

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.runtime.utils.telemetry_stub as tm


dynamic_dimension = np.ma.masked


def refer_to_faq_msg(question_num: int):
    try:
        t = tm.Telemetry()
        t.send_event('mo', 'error_info', "faq:" + str(question_num))
    except Exception:
        # Telemetry can be not initialized if it is used in MO IR Reader
        pass

    return '\n For more information please refer to Model Conversion API FAQ, question #{0}. ' \
           '(https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html' \
           '?question={0}#question-{0})'.format(question_num)


class NamedAttrsClass:
    def __init__(self, class_attrs: dict):
        for key, val in class_attrs.items():
            self.__setattr__(key, val)


def match_shapes(pattern: np.array, shape: np.array):
    """ Check if shape matches shape pattern handling undefined dimension and 0 in the pattern. """
    # Elements with value 0 and undefined values in pattern are just ignored. Other elements should match.
    if pattern.size != shape.size:
        return False
    indices = [i for i, n in enumerate(pattern) if n != 0 and n is not dynamic_dimension]
    return np.ma.allequal(pattern[indices], shape[indices])


def symm_match_shapes(shape1: np.array, shape2: np.array):
    """ Check if shape matches shape pattern handling -1 and 0 in the pattern. """
    # Elements with values -1 and 0 in both shapes are just ignored.
    # Other elements should match. Undefined elements can be one side only.
    return match_shapes(shape1, shape2) or match_shapes(shape2, shape1)


def deprecated_api(class_name=None, new_method_name=None):
    def deprecated(func):
        @functools.wraps(func)
        def deprecation_message(*args, **kwargs):
            dep_msg = "Call to deprecated function {}. ".format(func.__name__)
            if class_name is not None:
                dep_msg += "Please use {}.{} method" \
                           "".format(class_name.__name__ if not isinstance(class_name, str) else class_name,
                                     func.__name__ if new_method_name is None else new_method_name)
            warnings.warn(dep_msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return deprecation_message

    return deprecated


def array_to_str(node, attr):
    if not node.has_valid(attr):
        return None
    else:
        return ','.join(map(str, node[attr]))


def shrink_str_value(value: np.array, max_symbols=100):
    value = str(value)
    if len(value) > max_symbols:
        value = value.strip('\n')[:max_symbols - 3] + '...'
    return value


def files_by_pattern(dir: str, pattern: str, files_only=True, add_prefix=False):
    """
    Return a list of files and directories (or only files if the files_only is set to True) in the directory dir that
    match pattern string pattern.
    :param dir: Directory to search for files
    :param pattern: string defining pattern name
    :param files_only: flag to include only files (not directories) to the result
    :param add_prefix: flag to include the prefix string to the file names
    :return: list of file and directory names
    """
    pattern_compiled = re.compile(pattern)
    matched_file_names = []
    for file_name in os.listdir(dir):
        if re.match(pattern_compiled, file_name) and (not files_only or os.path.isfile(os.path.join(dir, file_name))):
            matched_file_names.append(os.path.join(dir, file_name) if add_prefix else file_name)
    return matched_file_names


def get_mo_root_dir():
    """
    Return the absolute path to the Model Conversion API root directory (where mo folder is located)
    :return: path to the MO root directory
    """
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(os.path.realpath(__file__))), os.pardir,
                                         os.pardir))


def group_by_with_binary_predicate(xs: list, predicate: Callable) -> list:
    """
    It is an analogue of the function group by from itertools, but with a binary predicate.
    In other words, group_by_with_binary_predicate generates a break or new group every time
    the value of the predicate function is False.
    :param xs: list of grouped value
    :param predicate: criterion of equality
    :return: grouped list
    """
    if not xs:
        return []
    prev = xs[0]
    sequence = [prev]
    result = []
    for x in xs[1:]:
        if predicate(prev, x):
            sequence.append(x)
            prev = x
        else:
            result.append(sequence)
            prev = x
            sequence = [prev]
    result.append(sequence)
    return result


def unique_by(xs: list, predicate: Callable) -> list:
    """
    This function groups elements of the list xs using 'predicate', and then takes one element from each group.
    :param xs: input list
    :param predicate: grouping criterion which is some binary predicate
    :return: list with unique elements
    """
    groups = group_by_with_binary_predicate(xs, predicate)
    return [group[0] for group in groups]


def check_values_equal(val1, val2):
    # This method is needed to check equality of values where some values can be None
    if val1 is None and val2 is None:
        return True
    if val1 is None:
        return False
    if val2 is None:
        return False
    return val1 == val2


np_map_cast = {bool: lambda x: bool_cast(x),
               np.int8: lambda x: np.int8(x),
               np.int16: lambda x: np.int16(x),
               np.int32: lambda x: np.int32(x),
               np.int64: lambda x: np.int64(x),
               np.uint8: lambda x: np.uint8(x),
               np.uint16: lambda x: np.uint16(x),
               np.uint32: lambda x: np.uint32(x),
               np.uint64: lambda x: np.uint64(x),
               np.float16: lambda x: np.float16(x),
               np.float32: lambda x: np.float32(x),
               np.double: lambda x: np.double(x),
               str: lambda x: str(x)}


def bool_cast(x):
    if isinstance(x, str):
        return False if x.lower() in ['false', '0'] else True if x.lower() in ['true', '1'] else 'unknown_boolean_cast'
    else:
        return bool(x)


def mo_array(value: Union[Iterable[Union[float, int]], float, int], dtype=None) -> np.ndarray:
    """
    This function acts in a same way as np.array except for the case when dtype is not provided
    and np.array return fp64 array this function returns fp32 array
    """
    x = np.array(value, dtype=dtype)
    if not isinstance(value, np.ndarray) and x.dtype == np.float64 and dtype != np.float64:
        x = x.astype(np.float32)
    return x


def validate_batch_in_shape(shape, layer_name: str):
    """
    Raises Error #39 if shape is not valid for setting batch size
    Parameters
    ----------
    shape: current shape of layer under validation
    layer_name: name of layer under validation
    """
    if len(shape) == 0 or (shape[0] is not dynamic_dimension and shape[0] not in (-1, 0, 1)):
        raise Error(('The input layer {} has a shape {} defined in the model. \n\n' +
                     'When you use "batch" option, Model Conversion API applies its value to the first ' +
                     'element of the shape if it is equal to -1, 0 or 1. Otherwise, this is the ambiguous ' +
                     'situation - it is not known in advance whether the layer has the batch ' +
                     'dimension or not.\n\n For example, you want to set batch dimension equals 100 ' +
                     'for the input layer "data" with shape (10,34). Although you can not use "batch", ' +
                     'you should pass "input_shape=[100,34]" instead of "batch=100". \n\n' +
                     'You can also specify batch dimension by setting "layout". \n\n')
                    .format(layer_name, shape))


def deduce_legacy_frontend_by_namespace(argv):
    if not hasattr(argv, 'framework') or not argv.framework:
        if getattr(argv, 'saved_model_dir', None) or getattr(argv, 'input_meta_graph', None):
            argv.framework = 'tf'
        elif getattr(argv, 'input_symbol', None) or getattr(argv, 'pretrained_model_name', None):
            argv.framework = 'mxnet'
        elif getattr(argv, 'input_proto', None):
            argv.framework = 'caffe'
        elif argv.input_model is None:
            raise Error('Path to input model is required: use "input_model".')
        else:
            argv.framework = guess_framework_by_ext(argv.input_model)

    return map(lambda x: argv.framework == x, ['tf', 'caffe', 'mxnet', 'kaldi', 'onnx'])


def guess_framework_by_ext(input_model_path: str) -> int:
    if re.match(r'^.*\.caffemodel$', input_model_path):
        return 'caffe'
    elif re.match(r'^.*\.pb$', input_model_path):
        return 'tf'
    elif re.match(r'^.*\.pbtxt$', input_model_path):
        return 'tf'
    elif re.match(r'^.*\.params$', input_model_path):
        return 'mxnet'
    elif re.match(r'^.*\.nnet$', input_model_path):
        return 'kaldi'
    elif re.match(r'^.*\.mdl', input_model_path):
        return 'kaldi'
    elif re.match(r'^.*\.onnx$', input_model_path):
        return 'onnx'
