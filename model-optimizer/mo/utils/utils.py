"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import functools
import warnings
import logging as log
import numpy as np


def refer_to_faq_msg(question_num: int):
    return '\n For more information please refer to Model Optimizer FAQ' \
           ' (<INSTALL_DIR>/deployment_tools/documentation/docs/MO_FAQ.html),' \
           ' question #{}. '.format(question_num)


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


def deprecated_api(class_name=None):
    def deprecated(func):
        @functools.wraps(func)
        def deprecation_message(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn on filter
            dep_msg = "Call to deprecated function {}. ".format(func.__name__)
            if class_name is not None:
                dep_msg += "Please use {}.{} method".format(class_name.__name__, func.__name__)
            warnings.warn(dep_msg, DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
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
