"""
 Copyright (c) 2018 Intel Corporation

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
    '''Check if shape matches shape pattern handling -1 and 0 in the pattern.'''
    # Elements with values -1 and 0 in pattern are just ignored.
    # Other elements should match.
    if pattern.size != shape.size:
        return False
    indices = [i for i, n in enumerate(pattern) if n not in [0, -1]]
    return np.array_equal(pattern[indices], shape[indices])
