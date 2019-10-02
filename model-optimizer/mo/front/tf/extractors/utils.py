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

from mo.front.tf.common import tf_data_type_decode
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def tf_tensor_shape(pb):
    return np.array([dim.size for dim in pb.dim], dtype=np.int64)


def tf_int_list(pb):
    return np.array(pb.i, dtype=np.int64)


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
    if len(shape) == 0:
        value = type_helper[1](pb_tensor)
        value = np.array(value).copy()
        assert len(value) == 1
        log.debug("value = {}, shape = {}, res = {}, res.shape = {}".format(str(type_helper[1](pb_tensor)), shape,
                                                                            np.array(type_helper[1](pb_tensor),
                                                                                     dtype=type_helper[0]),
                                                                            np.array(type_helper[1](pb_tensor),
                                                                                     dtype=type_helper[0]).shape))
        return np.array(value[0], dtype=type_helper[0])
        # return np.array(type_helper[1](pb_tensor), dtype=type_helper[0])
    else:
        if pb_tensor.tensor_content:
            flat = np.array(np.frombuffer(pb_tensor.tensor_content, type_helper[0]))
            if len(flat) == shape.prod():
                return flat.reshape(shape)
            else:
                log.warning("Shape and content size of tensor don't match, shape: {} content size: {}".
                            format(shape, len(flat)))
                # broadcast semantics: no reshape
                return flat
        else:
            # probably a broadcast semantics
            # load constant instead of tensor
            value = np.array(type_helper[1](pb_tensor), dtype=type_helper[0])
            log.warning("Broadcast of scalar to shape: {}".format(shape))
            return np.broadcast_to(value, shape=shape).copy()


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
