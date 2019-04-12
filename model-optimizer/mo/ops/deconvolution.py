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

from mo.front.common.partial_infer.utils import int64_array, float_array, mark_input_bins, assign_dims_to_weights, \
    tf_window_op_pad_infer
from mo.front.onnx.extractors.utils import get_backend_pad
from mo.front.extractor import spatial_getter
from mo.utils.error import Error
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Deconvolution(Op):
    op = 'Deconvolution'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return [
           'auto_pad',
           'group',
           ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
           ('kernel', lambda node: ','.join(map(str, node['kernel_spatial']))),

           ('pads_begin', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 0)))),
           ('pads_end', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 1)))),
           'output'
        ]

    def backend_attrs_v2(self):
        return [
            spatial_getter('stride-x', 'stride', 1),
            spatial_getter('stride-y', 'stride', 0),

            ('kernel-x', lambda node: node.kernel_spatial[1]),
            ('kernel-y', lambda node: node.kernel_spatial[0]),

            spatial_getter('pad-x', 'pad', 1, lambda x: x[0]),
            spatial_getter('pad-y', 'pad', 0, lambda x: x[0]),
            spatial_getter('pad-r', 'pad', 1, lambda x: x[1]),
            spatial_getter('pad-b', 'pad', 0, lambda x: x[1]),

            'auto_pad',
            'output',
            'group',
        ]


    @staticmethod
    def infer(node: Node):
        """
        Deconvolution has an input argument that explicitly determines output shape, so in contrast
        to the forward Conv2d we shouldn't infer output shape. We just use this output shape as
        an input shape and pass it to our utilities that computes numeric values for padding.
        They also deliver output shape that is interpreted here as input shape for convolution.
        We need to check that the real input shape and shape inferred by those utility functions match.
        """
        output_shape = np.array(node.in_node(0).value)
        kernel_shape = node.in_node(1).shape
        node['kernel_shape'] = kernel_shape
        if output_shape is None or kernel_shape is None or node.spatial_dims is None or node.stride is None:
            return

        if not node.has_valid('kernel_spatial_idx'):
            node['kernel_spatial_idx'] = np.delete([x for x in range(len(kernel_shape))], (node.input_feature_channel, node.output_feature_channel))

        spatial_dims = node.spatial_dims
        output_spatial = np.array(output_shape[spatial_dims])
        stride_spatial = np.array(node.stride[spatial_dims])
        node['kernel_spatial'] = np.array(kernel_shape[node.kernel_spatial_idx])
        node.pad_spatial_shape, input_spatial_for_check = tf_window_op_pad_infer(
            output_spatial, node.kernel_spatial, stride_spatial, node.auto_pad)

        assert all(input_spatial_for_check == node.in_node(2).shape[spatial_dims])

        pad = np.zeros((len(output_shape), 2), dtype=np.int64)
        pad[spatial_dims] = node.pad_spatial_shape
        node.pad = pad

        node.output = output_shape[node.channel_dims][0]
        node.output_shape = output_shape
        node.out_node().shape = output_shape

        mark_input_bins(node, ['weights'], 1)
        assign_dims_to_weights(node.in_node(1), node.kernel_spatial_idx, node.input_feature_channel,
                               node.output_feature_channel, len(kernel_shape))

        # cut shape input at port 0, it is already consumed
        node.graph.remove_edge(node.in_node(0).id, node.id)

        # reconnect input tensor from port 2 to port 0
        node.in_edge(2)['in'] = 0

        # OK, now we are sure this is a supported Deconvolution layer
        node.type = 'Deconvolution'
        node.op = 'Deconv2D'

        # Add permute_attrs
        PermuteAttrs.create_permute_attrs(node, attrs=[('pad', 'input:0'),
                                                       ('stride', 'input:0'),
                                                       ('output_shape', 'input:0'),
                                                       ('batch_dims', 'input:0'),
                                                       ('channel_dims', 'input:0'),
                                                       ('spatial_dims', 'input:0'),

                                                       ('kernel_shape', 'input:1'),
                                                       ('kernel_spatial_idx', 'input:1'),
                                                       ('input_feature_channel', 'input:1'),
                                                       ('output_feature_channel', 'input:1'),
                                                       ])

        PermuteAttrs.set_permutation(node.in_node(1), node,
                                     node.get_weights_permute if node.has_valid('get_weights_permute') else None)
