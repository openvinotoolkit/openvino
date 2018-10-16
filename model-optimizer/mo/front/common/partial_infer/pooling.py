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

from mo.front.common.partial_infer.utils import tf_window_op_pad_infer, copy_or_none


def tf_pool_infer(node, op=None):
    node.pad, node.out_node().shape = tf_window_op_pad_infer(copy_or_none(node.in_node().shape),
                                                             copy_or_none(node.window), copy_or_none(node.stride),
                                                             node.auto_pad)
    node.pad_spatial_shape = np.array(node.pad[node.spatial_dims]) if node.pad is not None else None


def pool_explicit_padding_infer(node):
    assert (len(node.in_nodes()) == 1)
    input_shape = node.in_node(0).shape
    if input_shape is None:
        return
    input_spatial_shape = input_shape[node.spatial_dims]

    if node.has_and_set('global_pool'):
        node.window[node.spatial_dims] = input_spatial_shape
    window_spatial_shape = node.window[node.spatial_dims]
    stride_spatial = node.stride[node.spatial_dims]
    assert any(stride_spatial), 'Stride can not be zero in node {}'.format(node.id)

    if node.has_valid('auto_pad'):
        node.pad_spatial_shape, node.output_spatial_shape = tf_window_op_pad_infer(input_spatial_shape, window_spatial_shape,
                                                                                   stride_spatial, node.auto_pad)
        pad = np.zeros((len(input_shape), 2), dtype=np.int64)
        pad[node.spatial_dims] = node.pad_spatial_shape
        node.pad = pad
    else:
        pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)
        rounding = np.floor
        if node.has_valid('pooling_convention') and node.pooling_convention == 'full':
            rounding = np.ceil
        output_spatial_shape = np.array(rounding(
            np.array(input_spatial_shape + pad_spatial_shape - window_spatial_shape, dtype=np.float) / stride_spatial),
            dtype=np.int64) + 1

        original_pads = np.array([i[1] for i in node.pad_spatial_shape])

        for i in range(len(input_spatial_shape)):
            if original_pads[i] and (output_spatial_shape[i] - 1) * stride_spatial[i] >= \
                    input_spatial_shape[i] + original_pads[i]:
                output_spatial_shape[i] -= 1

        node.output_spatial_shape = output_spatial_shape

    output_shape = input_shape.copy()
    output_shape[node.spatial_dims] = node.output_spatial_shape
    node.out_node().shape = output_shape
