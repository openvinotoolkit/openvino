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

import logging as log

import numpy as np


def int64_array(l: list):
    return np.array(l, dtype=np.int64)


def float_array(l: list):
    return np.array(l, dtype=np.int64)


def mark_input_bins(node, names=('weights', 'biases'), start_port: int = 1):
    """
    Preparing necessary attributes for edges at input ports starting from start_port.
    It is applicable for convolution and other operations that has constant inputs which
    are intended to be dumped as IE IR bin file.
    """
    nports = len(node.in_nodes())
    for i, name in enumerate(names):
        port = i + start_port
        if port >= nports:
            break
        if node.in_node(port).value is not None:
            node.in_edge(port)['bin'] = name


def assign_dims_to_weights(node, spatial, input_channel, output_channel=None, dims_number=None):
    if spatial is not None:
        node['spatial_dims'] = np.array(spatial, dtype=np.int64)
    node['input_channel_dim'] = np.array(input_channel, dtype=np.int64)
    node['output_channel_dim'] = np.array(output_channel, dtype=np.int64)
    if 'input_channel_dim' not in node['dim_attrs']:
        node['dim_attrs'].append('input_channel_dim')
    node['dims_number'] = dims_number


def copy_or_none(x):
    return x.copy() if x is not None else None


def convert_tf_padding_to_str(padding):
    mapping = {b'SAME': 'same_upper', b'VALID': 'valid'}
    return mapping[padding.s]


# TODO eliminate this dependency and pass necessary function as an argument
def tf_window_op_pad_infer(input, window, stride, auto_pad):
    if input is None or window is None or stride is None or auto_pad is None:
        return (None, None)
    if auto_pad in ['same_lower', 'same_upper']:
        if auto_pad == 'same_upper':
            output = np.int64(np.ceil(input / stride))
        else:
            output = np.int64(np.floor(input / stride))
        residual = input % stride
        mask = residual == 0
        full_pad = window.copy()
        full_pad[mask] -= stride[mask]
        mask = np.logical_not(mask)
        full_pad[mask] -= input[mask] % stride[mask]
        full_pad = np.maximum(full_pad, 0)
        low_pad = np.int64(full_pad / 2)
        high_pad = full_pad - low_pad
        pad = np.array([low_pad, high_pad]).transpose()
    elif auto_pad == 'valid':
        output = np.int64(np.ceil((input - window + 1) / stride))
        pad = np.zeros((len(output), 2), dtype=np.int64)
    else:
        log.error("Unsupported padding scheme: {}".format(auto_pad))
        pad = None
        output = None
    return (pad, output)
