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

nchw_to_nhwc_permute = np.array([0, 2, 3, 1], dtype=np.int64)
nhwc_to_nchw_permute = np.array([0, 3, 1, 2], dtype=np.int64)


def convert_shape(shape: np.array, permute: np.array):
    result = [0, 0, 0, 0]
    for ind, perm_ind in enumerate(permute):
        result[ind] = shape[perm_ind]
    return np.array(result)


def get_height_dim(layout: str):
    """
    Gets index of the dimension corresponding to height.
    :param layout: string representing layout: NCHW or NHWC usually.
    :return: index of the 'H' character
    """
    return layout.find('H')


def get_width_dim(layout: str):
    """
    Gets index of the dimension corresponding to width.
    :param layout: string representing layout: NCHW or NHWC usually.
    :return: index of the 'W' character
    """
    return layout.find('W')


def get_features_dim(layout: str):
    """
    Gets index of the dimension corresponding to features.
    :param layout: string representing layout: NCHW or NHWC usually.
    :return: index of the 'C' character
    """
    return layout.find('C')
