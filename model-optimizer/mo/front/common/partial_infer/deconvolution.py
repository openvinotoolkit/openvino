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

from mo.front.common.partial_infer.utils import mark_input_bins, assign_dims_to_weights, \
    tf_window_op_pad_infer


def tf_deconv2d_infer(node):
    """
    Deconvolution has an input argument that explicitly determines output shape, so in contrast
    to the forward Conv2d we shouldn't infer output shape. We just use this output shape as
    an input shape and pass it to our utilities that computes numeric values for padding.
    They also deliver output shape that is interpreted here as input shape for convolution.
    We need to check that the real input shape and shape inferred by those utility functions match.
    """
    output_shape = np.array(node.in_node(0).value)
    kernel_shape = node.in_node(1).shape
    if output_shape is None or kernel_shape is None or node.spatial_dims is None or node.stride is None:
        return
    spatial_dims = node.spatial_dims
    output_spatial = np.array(output_shape[spatial_dims])
    stride_spatial = np.array(node.stride[spatial_dims])
    kernel_spatial = np.array(kernel_shape[0:len(spatial_dims)])  # kernel spatial dims go first
    node.pad_spatial_shape, input_spatial_for_check = tf_window_op_pad_infer(
        output_spatial, kernel_spatial, stride_spatial, node.auto_pad)

    assert all(input_spatial_for_check == node.in_node(2).shape[spatial_dims])

    pad = np.zeros((len(output_shape), 2), dtype=np.int64)
    pad[spatial_dims] = node.pad_spatial_shape
    node.pad = pad

    node.output_shape = output_shape
    node.out_node().shape = output_shape

    mark_input_bins(node, ['weights'], 1)
    assign_dims_to_weights(node.in_node(1), [0, 1], [3], [2], 4)

    # cut shape input at port 0, it is already consumed
    node.graph.remove_edge(node.in_node(0).id, node.id)

    # reconnect input tensor from port 2 to port 0
    node.in_edge(2)['in'] = 0

    # OK, now we are sure this is a supported Deconvolution layer
    node.type = 'Deconvolution'
    node.op = 'Deconv2D'
