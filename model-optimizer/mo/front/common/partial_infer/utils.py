# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import Iterable, List, Union

import numpy as np


def int64_array(value: Union[Iterable[Union[float, int]], float, int]) -> np.ndarray:
    return np.array(value, dtype=np.int64)


def float_array(value: Union[Iterable[Union[float, int]], float, int]) -> np.ndarray:
    return np.array(value, dtype=np.float64)


def float32_array(value: Union[Iterable[Union[float, int]], float, int]) -> np.ndarray:
    return np.array(value, dtype=np.float32)


def mark_input_bins(node, names=('weights', 'biases'), start_port: int = 1):
    """
    Preparing necessary attributes for edges at input ports starting from start_port.
    It is applicable for convolution and other operations that have constant inputs which
    are intended to be dumped as IE IR bin file.
    """
    for port, name in enumerate(names, start=start_port):
        if port in node.in_nodes() and node.in_node(port).has_valid('value'):
            node.in_edge(port)['bin'] = name


def assign_dims_to_weights(node, spatial, input_channel, output_channel=None, dims_number=None):
    if spatial is not None:
        node['spatial_dims'] = np.array(spatial, dtype=np.int64)
    node['input_channel_dim'] = np.array(input_channel, dtype=np.int64)
    node['output_channel_dim'] = np.array(output_channel, dtype=np.int64)
    if 'dim_attrs' in node and 'input_channel_dim' not in node['dim_attrs']:
        node['dim_attrs'].append('input_channel_dim')
    node['dims_number'] = dims_number


def copy_or_none(x):
    return x.copy() if x is not None else None


def convert_tf_padding_to_str(padding):
    mapping = {'SAME': 'same_upper', 'VALID': 'valid'}
    return mapping[padding]


def convert_deconv_tf_padding_to_str(padding):
    # according to the formulas for calculating "auto_pad" values of the
    # ConvBackpropData layer in the Operation Specification,
    # the "same_lower" value matches to the "same" value for conv_transpose layer in TensorFlow
    mapping = {'SAME': 'same_lower', 'VALID': 'valid'}
    return mapping[padding]


# TODO eliminate this dependency and pass necessary function as an argument
def tf_window_op_pad_infer(input, window, stride, auto_pad, is_deconv=False):
    if input is None or window is None or stride is None or auto_pad is None:
        return (None, None)

    normalized_stride = stride
    if is_deconv:
        normalized_stride = 1 / stride

    if auto_pad in ['same_lower', 'same_upper']:
        output = np.int64(np.ceil(input / normalized_stride))
        residual = input % stride
        mask = residual == 0
        full_pad = window.copy()
        full_pad[mask] -= stride[mask]
        mask = np.logical_not(mask)  # pylint: disable=assignment-from-no-return
        full_pad[mask] -= input[mask] % stride[mask]
        full_pad = np.maximum(full_pad, 0)  # pylint: disable=assignment-from-no-return
        low_pad = np.int64(full_pad / 2)
        high_pad = full_pad - low_pad
        pad = np.array([low_pad, high_pad]).transpose()
    elif auto_pad == 'valid':
        output = np.int64(np.ceil((input - window + 1) / normalized_stride))
        pad = np.zeros((len(output), 2), dtype=np.int64)
    else:
        log.error("Unsupported padding scheme: {}".format(auto_pad))
        pad = None
        output = None
    return (pad, output)


def broadcast_shape(first_shape, second_shape):
    """
    Perform broadcasting of one shape to another for different shapes
    """
    shape = first_shape if len(first_shape) > len(second_shape) else second_shape
    new_shape = int64_array(shape)
    for i in range(len(shape)):
        a_val = first_shape[-i - 1] if i < len(first_shape) else 1
        b_val = second_shape[-i - 1] if i < len(second_shape) else 1
        assert a_val == 1 or b_val == 1 or a_val == b_val, "Input shape do not broadcast"
        new_val = b_val if a_val == 1 else a_val
        new_shape[-i - 1] = new_val
    return int64_array(new_shape)


def get_shape_from_slice(input_shape: np.ndarray, slices: List) -> np.ndarray:
    """
    Calculate shape of a tensor after slicing without actually creating the resulting tensor.
    Is introduced to prevent potentially large memory consumption.
    """
    output_shape = []
    num_new_axes = np.count_nonzero(list(map(lambda x: x is np.newaxis, slices)))
    num_ellipsis_inserts = len(input_shape) - len(slices) + num_new_axes + 1

    in_idx = 0
    for i, s in enumerate(slices):
        if isinstance(s, slice):
            output_shape.append(len(range(*s.indices(input_shape[in_idx]))))
            in_idx += 1
        elif s is np.newaxis:
            output_shape.append(1)
        elif type(s) in [int, np.int, np.int32, np.int64]:  # shrink_axis
            in_idx += 1
        elif s is Ellipsis:
            for idx in range(num_ellipsis_inserts):
                output_shape.append(input_shape[in_idx])
                in_idx += 1
        else:
            raise Exception('Element type of a slice List is unacceptable. '
                            'Allowed types are: Ellipsis, slice, int, and None. Instead got: '. format(type(s)))
    for i in range(in_idx, len(input_shape)):
        output_shape.append(input_shape[i])
    return int64_array(output_shape)
