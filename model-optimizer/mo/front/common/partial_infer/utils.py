# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import Iterable, List, Union

import numpy as np

dynamic_dimension = np.ma.masked
dynamic_dimension_value = -1000000007


def shape_array(value, dtype=np.int64, dyn_value=dynamic_dimension_value):
    # if the input already have masked values then we need to explicitly convert to dynamic_dimension_value and create
    # masked array from scratch, because otherwise method masked_equal will convert masked elements to "nan" value
    new_value = [item if item is not dynamic_dimension else dynamic_dimension_value for item in value]
    return np.ma.masked_equal(new_value, dynamic_dimension_value).astype(dtype=dtype)


def compare_dimensions(dim1, dim2):
    """
    Compare if dim1 is equal to dim2 or any of them is dynamic
    :param dim1: dimension to compare
    :param dim2: dimension to compare
    :return: boolean result of the comparison
    """
    return dim1 is dynamic_dimension or dim2 is dynamic_dimension or dim1 == dim2


def compare_shapes(shape1, shape2):
    if shape1.ndim != shape2.ndim:
        return False
    for d1, d2 in zip(shape1, shape2):
        if not compare_dimensions(d1, d2):
            return False
    return True


def unmask_shape(value):
    if not isinstance(value, np.ma.masked_array):
        return value
    else:
        return value.tolist(-1)


def is_fully_defined(value):
    if isinstance(value, np.ma.masked_array):
        return not np.ma.is_masked(value)
    elif isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value is not dynamic_dimension
        return np.all([item is not dynamic_dimension for item in value])
    elif isinstance(value, list):
        return True
    elif value is dynamic_dimension:
        return False
    return True


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
        return None, None

    normalized_stride = stride
    if is_deconv:
        normalized_stride = 1 / stride

    if auto_pad in ['same_lower', 'same_upper']:
        output = shape_array(np.ma.ceil(input / normalized_stride))
        residual = input % stride
        mask = residual == 0
        full_pad = window.copy()
        full_pad[mask] -= stride[mask]
        mask = np.logical_not(mask)  # pylint: disable=assignment-from-no-return
        full_pad[mask] -= input[mask] % stride[mask]
        full_pad = np.ma.maximum(full_pad, 0)  # pylint: disable=assignment-from-no-return
        low_pad = np.int64(full_pad / 2)
        high_pad = full_pad - low_pad
        pad = shape_array([low_pad, high_pad]).transpose()
    elif auto_pad == 'valid':
        output = np.int64(np.ceil((input - window + 1) / normalized_stride))
        pad = np.zeros((len(output), 2), dtype=np.int64)
    else:
        log.error("Unsupported padding scheme: {}".format(auto_pad))
        pad = None
        output = None
    return pad, output


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
        # TODO this looks like a hack that we check dynamic_dimension_value
        if s is dynamic_dimension or s == dynamic_dimension_value:
            output_shape.append(dynamic_dimension_value)
            in_idx += 1
        elif isinstance(s, slice):
            if input_shape[in_idx] is not dynamic_dimension:
                output_shape.append(len(range(*s.indices(input_shape[in_idx]))))
            else:
                output_shape.append(dynamic_dimension_value)
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
            raise Exception('Element type of a slice list is unacceptable: "{}"'.format(type(s)))
    for i in range(in_idx, len(input_shape)):
        output_shape.append(input_shape[i])
    return shape_array(output_shape)
