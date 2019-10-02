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

import logging as log

import numpy as np

from mo.utils.error import Error

nchw_to_nhwc_permute = np.array([0, 2, 3, 1], dtype=np.int64)
nhwc_to_nchw_permute = np.array([0, 3, 1, 2], dtype=np.int64)
supported_layouts = ('NCHW', 'NHWC')
# the attribute 'layout' in the graph.graph can have two values only: "NCHW" or "NHWC". If the tensor has 5 dimensions
# then it is necessary to transform "NCHW" to "NCDHW" and "NHWC" to "NDHWC" respectively. The dictionary below id used
# for this purpose.
indices_mapping = {4: {'NCHW': 'NCHW',
                       'NHWC': 'NHWC'},
                   5: {'NCHW': 'NCDHW',
                       'NHWC': 'NDHWC'}}


def convert_shape(shape: np.array, permute: np.array):
    result = [0, 0, 0, 0]
    for ind, perm_ind in enumerate(permute):
        result[ind] = shape[perm_ind]
    return np.array(result)


def get_depth_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to depth.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'D' character
    """
    assert layout in supported_layouts
    assert shape_len == 5
    return indices_mapping[shape_len][layout].find('D')


def get_height_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to height.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'H' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('H')


def get_width_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to width.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'W' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('W')


def get_features_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to features.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'C' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('C')


def get_batch_dim(layout: str, shape_len: int):
    """
    Gets index of the dimension corresponding to batch.
    :param layout: string representing layout: NCHW or NHWC usually.
    :param shape_len: the shape length.
    :return: index of the 'N' character
    """
    assert layout in supported_layouts
    assert 4 <= shape_len <= 5
    return indices_mapping[shape_len][layout].find('N')


def shape_for_layout(layout: str, **kwargs):
    """
    Creates 4D or 5D tensor with the layout with specified dimension sizes.
    :param layout: layout string.
    :param kwargs: dictionary that contains the dimension sizes using the following keys: 'batch', 'features', 'depth',
    'height', 'width'.
    :return: np.array of type np.int64 with 4 or 5 elements.
    """
    assert layout in supported_layouts
    for required_key in ('batch', 'features', 'height', 'width'):
        if required_key not in kwargs:
            raise Error('Required parameter "{}" is missing.'.format(required_key))
    for key in kwargs.keys():
        if key not in ('batch', 'features', 'height', 'width', 'depth'):
            raise Error('Parameter "{}" is not supported.'.format(key))

    depth = kwargs.get('depth', None)
    shape_len = 4 + (depth is not None)
    output_shape = np.ones(shape=[shape_len], dtype=np.int64)
    output_shape[get_batch_dim(layout, shape_len)] = kwargs['batch']
    output_shape[get_height_dim(layout, shape_len)] = kwargs['height']
    output_shape[get_width_dim(layout, shape_len)] = kwargs['width']
    output_shape[get_features_dim(layout, shape_len)] = kwargs['features']
    if depth is not None:
        output_shape[get_depth_dim(layout, shape_len)] = depth
    return output_shape
