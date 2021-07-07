# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import re
import warnings

from typing import Callable

import numpy as np


def refer_to_faq_msg(question_num: int):
    return '\n For more information please refer to Model Optimizer FAQ, question #{0}. ' \
           '(https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ.html' \
           '?question={0}#question-{0})'.format(question_num)


class NamedAttrsClass:
    def __init__(self, class_attrs: dict):
        for key, val in class_attrs.items():
            self.__setattr__(key, val)


def match_shapes(pattern: np.array, shape: np.array):
    """ Check if shape matches shape pattern handling -1 and 0 in the pattern. """
    # Elements with values -1 and 0 in pattern are just ignored.
    # Other elements should match.
    if pattern.size != shape.size:
        return False
    indices = [i for i, n in enumerate(pattern) if n not in [0, -1]]
    return np.array_equal(pattern[indices], shape[indices])


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
    Return the absolute path to the Model Optimizer root directory (where mo.py file is located)
    :return: path to the MO root directory
    """
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(os.path.realpath(__file__))), os.pardir,
                                         os.pardir))


def group_by_with_binary_predicate(xs: list, predicate: Callable) -> list:
    """
    It is an analogue of the function groupby from itertools, but with a binary predicate.
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
