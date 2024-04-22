# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array, int64_array
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value
from openvino.tools.mo.front.tf.common import tf_data_type_decode
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def tf_tensor_shape(pb):
    return shape_array([dim.size if dim.size >= 0 else dynamic_dimension_value for dim in pb.dim])


def tf_int_list(pb):
    return int64_array(pb.i)


def tf_dtype_extractor(pb_dtype, default=None):
    return tf_data_type_decode[pb_dtype][0] if pb_dtype in tf_data_type_decode else default


def tf_data_format_spatial(pb):
    if b"DHW" in pb.s:
        return [pb.s.index(c) for c in b"DHW"]
    return [pb.s.index(c) for c in b"HW"]


def tf_data_format_channel(pb):
    return [pb.s.index(b'C')]


def tf_data_format_batch(pb):
    return [pb.s.index(b'N')]


def get_tf_node_port(tensor):
    delim = ':'
    # tensor should have form 'name:port' or just 'name'
    name_parts = tensor.split(delim)
    if len(name_parts) == 1:
        # just 'name', then port is 0 by default
        return name_parts[0], 0
    else:
        # 'name:port', note name can contain ':' also but port is the last part
        # TODO Is 'name' that contains other ':'s considered valid by TF?
        return delim.join(name_parts[:-1]), int(name_parts[-1])


def tf_tensor_content(tf_dtype, shape, pb_tensor):
    type_helper = tf_data_type_decode[tf_dtype] if tf_dtype in tf_data_type_decode else None
    if type_helper is None:
        raise Error("Data type is unsupported: {}. " +
                    refer_to_faq_msg(50), tf_dtype)

    decode_err_msg = 'Failed to parse a tensor with Unicode characters. Note that OpenVINO does not support ' \
                     'string literals, so the string constant should be eliminated from the graph.'
    if pb_tensor.tensor_content:
        value = mo_array(np.frombuffer(pb_tensor.tensor_content, type_helper[0]))
    else:
        # load typed value
        if type_helper[0] != str:
            value = mo_array(type_helper[1](pb_tensor), dtype=type_helper[0])
        else:
            try:
                value = mo_array(type_helper[1](pb_tensor), dtype=type_helper[0])
            except UnicodeDecodeError:
                log.error(decode_err_msg, extra={'is_warning': True})
                value = mo_array(type_helper[1](pb_tensor))

    # Ignore an empty value, if len(shape) > 1
    # For example, value = [] and shape = [1, 1, 0]
    # This is needed to reshape this value later and to return reshaped value = [[[]]]
    # Otherwise there can be failures during partial inference, because we are storing an empty value with incorrect
    # shape
    if len(shape) == 0 or (len(shape) == 1 and shape.prod() == 0):
        try:
            value_length = len(value)
        except TypeError:
            # case, when value is a scalar
            return value
        if value_length == 1:
            # return scalar if shape is [] otherwise broadcast according to shape
            try:
                return mo_array(value[0], dtype=type_helper[0])
            except UnicodeDecodeError:
                log.error(decode_err_msg, extra={'is_warning': True})
                return mo_array(value[0])
        else:
            if len(shape) == 0 and value_length == 0:
                # Since TF 2.10 the model freezing can produce constants with non-empty tensor
                # but with undefined value []
                # in this case, the tensor is filled with the default value
                # that is 0 for numeric types and "" for string
                default_value = 0 if type_helper[0] != str else ""
                value = mo_array(default_value, dtype=type_helper[0])
            # no shape, return value as is
            return value

    if len(value) != shape.prod():
        log.warning("Shape and content size of tensor don't match, shape: {} content size: {}".
                    format(shape, len(value)))

        if len(value) == 0:
            # Since TF 2.10 the model freezing can produce constants with non-empty tensor but with undefined value []
            # In this case, the tensor is filled with the default value that is 0 for numeric types and "" for string
            default_value = 0 if type_helper[0] != str else ""
            value_flatten = mo_array([default_value], dtype=type_helper[0])
        else:
            value_flatten = value.flatten()

        # broadcast semantics according to TensorFlow v1.5 documentation:
        # The argument value can be a constant value, or a list of values of type dtype. If value is a list,
        # then the length of the list must be less than or equal to the number of elements implied by the shape
        # argument (if specified). In the case where the list length is less than the number of elements specified
        # by shape, the last element in the list will be used to fill the remaining entries.
        add_value = value_flatten[-1]
        add_length = shape.prod() - len(value_flatten)
        value = np.concatenate([value_flatten, np.full([add_length], add_value)])

    return value.reshape(shape)


def check_attr_type(a):
    """
      Check type of attribute from TF prototxt message
      param: a - attribute from TF prototxt message
      return: type of attribute
    """
    if a.s:
        return 's'
    if a.i:
        return 'i'
    if a.f:
        return 'f'
    if a.b:
        return 'b'
    if a.type:
        return 'type'
    if a.shape and a.shape.dim:
        return 'shape'
    if a.list:
        return 'list'


def collect_tf_attrs(attrs):
    """
     Function generates map for attributes and parsing functions
     param: attrs  - TF proto message with attributes
     return: mapping attributes and parsing functions ready for use in update_node_stat function
    """
    ret_attrs = {}
    type_parsers = {
        's': lambda x: x.s,
        'i': lambda x: x.i,
        'f': lambda x: x.f,
        'b': lambda x: x.b,
        'type': lambda x: tf_dtype_extractor(x.type),
        'shape': lambda x: tf_tensor_shape(x.shape),
        'list': lambda x: x.list
    }

    for a in attrs:
        t = check_attr_type(attrs[a])
        a_l = attrs[a]
        while t == 'list':
            a_l = type_parsers[t](attrs[a])
            t = check_attr_type(a_l)

        ret_attrs[a] = type_parsers[t](a_l)

    return ret_attrs
