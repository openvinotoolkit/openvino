"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array


def make_equal_rank(shape_1: np.array, shape_2: np.array):
    """
    Prepend shape with smaller length with 1. Return updates shapes
    :param shape_1: first shape
    :param shape_2: second shape
    :return: tuple with updated shapes
    """
    while len(shape_1) < len(shape_2):
        shape_1 = np.insert(shape_1, 0, 1)

    while len(shape_2) < len(shape_1):
        shape_2 = np.insert(shape_2, 0, 1)

    return shape_1, shape_2


def uni_directional_shape_broadcasting(input_shape: np.array, target_shape: np.array):
    """
    Uni-directional broadcasting of two shapes following the numpy semantic
    :param input_shape: input shape to broadcast
    :param target_shape: target shape
    :return: broadcasted shape or None if broadcasting cannot be performed
    """
    input = input_shape.copy()

    # in one-directional broadcasting the target shape rank can be higher or equal than input shape
    if len(input_shape) > len(target_shape):
        log.debug('The shape "{}" cannot be broadcasted to "{}"'.format(input_shape, target_shape))
        return None

    # prepend input shape with 1s
    input, target_shape = make_equal_rank(input, target_shape)

    for left, right in zip(input, target_shape):
        if left != right and left != 1:
            log.debug('The shape "{}" cannot be broadcasted to "{}"'.format(input_shape, target_shape))
            return None

    return target_shape


def bi_directional_shape_broadcasting(input_shape_1: np.array, input_shape_2: np.array):
    """
    Bi-directional broadcasting of two shapes following numpy semantic
    :param input_shape_1: first shape to broadcast
    :param input_shape_2: second shape to broadcast
    :return: broadcasted shape or None if broadcasting cannot be performed
    """
    shape_1 = input_shape_1.copy()
    shape_2 = input_shape_2.copy()
    shape_1, shape_2 = make_equal_rank(shape_1, shape_2)

    for left, right in zip(shape_1, shape_2):
        if left != right and left != 1 and right != 1:
            log.debug('The shape "{}" cannot be broadcasted to "{}"'.format(input_shape_1, input_shape_2))
            return None

    return np.maximum(shape_1, shape_2)


def explicit_shape_broadcasting(input_shape: np.array, target_shape: np.array, axes_mapping: np.array) -> np.array:
    """
    Explicit shape broadcasting of input tensor. Function only asserts that values are correct and normalizes axes.
    Resulting shape is equal to target_shape.
    :param input_value: input value to broadcast
    :param target_shape: target shape
    :param axes_mapping: a list of axis indices, each index maps an axis from the input_value to axis in the output
    :return: broadcasted shape and normalized axes
    """
    assert np.all(np.diff(axes_mapping) >= 0), "axes_mapping is not sorted"
    assert len(axes_mapping) == len(input_shape), "size of axes_mapping does not match to rank of input"
    axes_mapping = np.array(list(map(lambda axis: axis + len(target_shape) if axis < 0 else axis, axes_mapping)))

    res = target_shape.copy()
    for i, axis in enumerate(axes_mapping):
        assert 0 <= axis < len(res), "axis value from axes_mapping exceeds rank of target_shape"
        assert res[axis] == input_shape[i], "specified mapping axis in target_shape differs from axis in input_shape"
    return res, axes_mapping


def uni_directional_broadcasting(input_value: np.array, target_shape: np.array):
    """
    Uni-directional broadcasting of input tensor to target shape following the numpy semantic
    :param input_value: input value to broadcast
    :param target_shape: target shape
    :return: broadcasted value
    """
    assert uni_directional_shape_broadcasting(int64_array(input_value.shape), target_shape) is not None, \
        'The tensor of shape "{}" cannot be uni-directionally broadcasted to shape "{}"'.format(input_value.shape,
                                                                                                target_shape)
    return np.broadcast_to(input_value, target_shape)


def bi_directional_broadcasting(input_value: np.array, second_shape: np.array):
    """
    Bi-directional broadcasting of input tensor to target shape following the numpy semantic
    :param input_value: input value to broadcast
    :param second_shape: second tensor shape
    :return: broadcasted value
    """
    assert bi_directional_shape_broadcasting(int64_array(input_value.shape), second_shape) is not None, \
        'The tensor of shape "{}" cannot be bi-directionally broadcasted to shape "{}"'.format(input_value.shape,
                                                                                               second_shape)
    return np.array(input_value * np.ones(second_shape), dtype=input_value.dtype)


def explicit_broadcasting(input_value: np.array, target_shape: np.array, axes_mapping: np.array) -> np.array:
    """
    Explicit broadcasting of input tensor. Resulting shape is equal to target_shape except for axes specified in axes_mapping
    :param input_value: input value to broadcast
    :param target_shape: target shape
    :param axes_mapping: a list of axis indices, each index maps an axis from the input_value to axis in the output
    :return: broadcasted value
    """
    res_shape, normalized_axes_mapping = explicit_shape_broadcasting(input_value.shape, target_shape, axes_mapping)
    expand_dim_axis = set(np.arange(len(target_shape))) - set(normalized_axes_mapping)

    input_expanded = np.expand_dims(input_value.copy(), axis=list(expand_dim_axis))
    return np.broadcast_to(input_expanded, res_shape)
