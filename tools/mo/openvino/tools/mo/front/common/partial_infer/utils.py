# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import Iterable, List, Union

import numpy as np

from openvino.tools.mo.utils.error import Error

dynamic_dimension = np.ma.masked
# numpy masked array for integer values forces us to select one integer number to be considered as a missing/invalid
# value. Since the primary purpose of usage of masked arrays in the MO is to specify dynamic dimension, the big prime
# (by modulo) negative number is selected as such a value
dynamic_dimension_value = -1000000007


def shape_array(value, dtype=np.int64):
    # if the input tensor has masked values then they should be explicitly converted to dynamic_dimension_value and
    # a masked array should be created from scratch, otherwise, method "masked_equal" will convert masked elements to
    # "nan" values
    if isinstance(value, Iterable) and (not isinstance(value, np.ndarray) or value.ndim != 0):
        value = [item if item is not dynamic_dimension else dynamic_dimension_value for item in value]
    return np.ma.masked_equal(value, dynamic_dimension_value).astype(dtype=dtype)


def undefined_shape_of_rank(rank: int):
    """
    Create a shape of specified rank with all dynamic dimensions.

    :param rank: requested rank of the output shape
    :return: shape array
    """
    return shape_array([dynamic_dimension_value] * rank)


def compatible_dims(dim1, dim2):
    """
    Compare if dim1 is equal to dim2 or any of them is dynamic

    :param dim1: dimension to compare
    :param dim2: dimension to compare
    :return: boolean result of the comparison
    """
    return dim1 is dynamic_dimension or dim2 is dynamic_dimension or dim1 == dim2


def compatible_shapes(shape1, shape2):
    """
    Compares two shape tensors. The shapes are considered equal if they have the same rank and the corresponding
    dimensions are either equal or at least one of them is dynamic.

    :param shape1: the first shape to compare
    :param shape2: the second shape to compare
    :return: boolean result of the comparison
    """
    if shape1.ndim != shape2.ndim:
        return False
    if shape1.size != shape2.size:
        return False
    for d1, d2 in zip(shape1, shape2):
        if not compatible_dims(d1, d2):
            return False
    return True


def strict_compare_tensors(tensor1, tensor2):
    """
    Strict comparison of two tensors. The tensors are equal iff their corresponding elements are equal or both are
    dynamic.

    :param tensor1: the first tensor to compare
    :param tensor2: the second tensor to compare
    :return: boolean result of the comparison
    """
    if tensor1 is None and tensor2 is None:
        return True
    if tensor1 is None or tensor2 is None:
        return False

    if not isinstance(tensor1, np.ma.masked_array):
        tensor1 = shape_array(tensor1)
    if not isinstance(tensor2, np.ma.masked_array):
        tensor2 = shape_array(tensor2)

    if tensor1.ndim != tensor2.ndim:
        return False
    if tensor1.size != tensor2.size:
        return False
    if tensor1.ndim == 0:
        return tensor1.item() == tensor2.item()
    if not np.array_equal(tensor1.shape, tensor2.shape):
        return False
    for d1, d2 in zip(tensor1.flatten(), tensor2.flatten()):
        if (d1 is not dynamic_dimension) ^ (d2 is not dynamic_dimension):
            return False
        elif d1 is not dynamic_dimension and d1 != d2:
            return False
    return True


def shape_delete(shape: np.ma.masked_array, obj: [int, list]):
    """
    Removes element in the input tensor shape (presumably the numpy masked array) specified by index/indices obj.
    The function is implemented to avoid usage of np.delete which corrupts information about the masked elements.

    :param shape: the shape object to remove elements from
    :param obj: the list or a single integer defining index(es) of elements to remove
    :return: shape with removed selected elements
    """
    if isinstance(obj, (int, np.int64, np.int32)):
        return shape_delete(shape, [obj])
    elif isinstance(obj, np.ndarray):
        return shape_delete(shape, obj.tolist())
    elif isinstance(obj, list):
        result = shape.copy()
        obj = [item if item >= 0 else len(shape) + item for item in obj]
        for index in sorted(obj, reverse=True):
            assert 0 <= index < len(result), 'Incorrect element index {} to remove from {}'.format(index, result)
            result = np.ma.concatenate((result[:index], result[index + 1:]))
        return result
    else:
        raise Error('Incorrect parameter type of "obj": {}'.format(type(obj)))


def shape_insert(shape: [np.ndarray, list], pos: int, obj: [int, list, np.ndarray, dynamic_dimension]):
    """
    Insert element(s) in the input tensor shape (presumably the numpy masked array) specified by position pos.
    The function is implemented to avoid usage of np.insert which corrupts information about the masked elements.

    :param shape: the shape object to insert element(s) to
    :param pos: the position to insert the elements into
    :param obj: the list or a single integer or the dynamic_dimension_value or numpy array to insert
    :return: shape with inserted elements
    """
    if isinstance(obj, (int, np.int64, np.int32)) or obj is dynamic_dimension_value:
        return shape_insert(shape, pos, [obj])
    elif isinstance(obj, (np.ndarray, list)):
        return np.ma.concatenate((shape_array(shape[:pos]), shape_array(obj), shape_array(shape[pos:])))
    else:
        raise Error('Incorrect parameter type of "obj": {}'.format(type(obj)))


def unmask_shape(value: [np.ma.masked_array, np.array]):
    """
    Converts all dynamic_dimension values from the input tensor to -1. Used to generate shapes for the IR.

    :param value: the value to be unmasked.
    :return: the value where dynamic_dimension elements are converted to -1.
    """
    if not isinstance(value, np.ma.masked_array):
        return value
    else:
        return value.tolist(-1)


def is_fully_defined(value):
    """
    Checks that provided input tensor is fully defined. The input value can be of different types: scalar, list, array,
    masked array.

    :param value: the value to check
    :return: the result of the check
    """
    if value is None:
        return False
    elif isinstance(value, np.ma.masked_array):
        return not np.ma.is_masked(value)
    elif isinstance(value, np.ndarray):  # numpy array cannot contain dynamic values
        return True
    elif isinstance(value, list) or isinstance(value, tuple):
        return dynamic_dimension not in value
    elif value is dynamic_dimension:
        return False
    return True


def int64_array(value: Union[Iterable[Union[float, int]], float, int]) -> np.ndarray:
    return np.array(value, dtype=np.int64)


def float32_array(value: Union[Iterable[Union[float, int]], float, int]) -> np.ndarray:
    return np.array(value, dtype=np.float32)


def int8_array(value: Union[Iterable[Union[float, int]], float, int]) -> np.ndarray:
    return np.array(value, dtype=np.int8)


def float_array(value: Union[Iterable[Union[float, int]], float, int]) -> np.ndarray:
    return float32_array(value)


def mo_array(value: Union[Iterable[Union[float, int]], float, int], dtype=None) -> np.ndarray:
    """
    This function acts in a same way as np.array except for the case when dtype is not provided
    and np.array return fp64 array this function returns fp32 array
    """
    x = np.array(value, dtype=dtype)
    if not isinstance(value, np.ndarray) and x.dtype == np.float64 and dtype != np.float64:
        x = x.astype(np.float32)
    return x


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
        node['spatial_dims'] = int64_array(spatial)
    node['input_channel_dim'] = int64_array(input_channel)
    node['output_channel_dim'] = int64_array(output_channel)
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
def tf_window_op_pad_infer(input, window, stride, auto_pad, is_deconv=False, dilation=None):
    if input is None or window is None or stride is None or auto_pad is None:
        return None, None

    if dilation is None:
        dilation = np.ones(len(input), dtype=np.int64)

    normalized_stride = stride
    if is_deconv:
        normalized_stride = 1 / stride

    if auto_pad in ['same_lower', 'same_upper']:
        output = np.ma.ceil(input / normalized_stride)
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
        output = np.int64(np.ceil((input - ((window - 1) * dilation + 1) + 1) / normalized_stride))
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
        if s is dynamic_dimension or s == dynamic_dimension_value:
            output_shape.append(dynamic_dimension_value)
            in_idx += 1
        elif isinstance(s, slice):
            if input_shape[in_idx] is not dynamic_dimension and not is_dynamic_slice(s):
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


def is_dynamic_slice(s: [slice, int, None]):
    """
    The function checks that the specified slice produces dynamic value.
    :param s: slice object
    :return: the result of the check
    """
    return isinstance(s, slice) and (s.start is dynamic_dimension or
                                     s.stop is dynamic_dimension or
                                     s.step is dynamic_dimension)


def reverse_bypass_infer(node, in_ports: List[int]):
    """
    Copies shapes from the out_port 0 into ports specified in the in_ports


    :param node:
    :param in_ports: input ports for which shape will be updated
    :return:
    """
    # WA: for cases when terminal Identity node has only output control dependency edges
    # For this case the graph is not correctly build because the Identity node goes
    # without Result node
    if node.out_port(0).disconnected():
        return

    output_shape = node.out_port(0).data.get_shape()
    if output_shape is not None:
        for port in in_ports:
            assert node.is_in_port_connected(port)
            if node.in_port(port).data.get_shape() is None:
                node.in_port(port).data.set_shape(output_shape)


def clarify_partial_shape(shapes: List):
    """
    returns more precise partial shape from a set of partial shapes,
    e.g. pshape_1 = [dyn, 2, dyn], pshape_2 = [10, dyn, dyn] => out_shape = [10, 2, dyn]
    :param shapes:
    :return:
    """
    assert len(shapes) > 0
    out_shape = shapes[0]
    for shape in shapes:
        assert compatible_shapes(shape, out_shape), "shapes {} and {} are not compatible".format(
            unmask_shape(shape), unmask_shape(out_shape))
        shape_unmasked = shape.data.copy()
        for i, dim in enumerate(shape_unmasked):
            if dim != dynamic_dimension_value:
                out_shape[i] = dim
    return out_shape


def set_input_shapes(node, *shapes: List):
    assert len(shapes) <= len(node.in_ports())
    
    for i, shape in enumerate(shapes):
        if node.is_in_port_connected(i) and node.in_port(i).data.get_shape() is None:
            node.in_port(i).data.set_shape(shape)
