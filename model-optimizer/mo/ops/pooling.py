"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.front.common.partial_infer.utils import tf_window_op_pad_infer, int64_array, float_array
from mo.front.onnx.extractors.utils import get_backend_pad
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs
from mo.utils.error import Error
from mo.front.extractor import bool_to_str


class Pooling(Op):
    op = 'Pooling'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return [
            ('strides', lambda node: ','.join(map(str, node['stride'][node.spatial_dims]))),
            ('kernel', lambda node: ','.join(map(str, node['window'][node.spatial_dims]))),

            ('pads_begin', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 0)))),
            ('pads_end', lambda node: ','.join(map(str, get_backend_pad(node.pad, node.spatial_dims, 1)))),

            ('exclude-pad', lambda node: bool_to_str(node, 'exclude_pad')),

            'rounding_type',
            ('auto_pad', lambda node: node.auto_pad if node.has_valid('auto_pad') else 'explicit'),
        ]

    @staticmethod
    def infer(node: Node):
        assert (len(node.in_nodes()) == 1)
        input_shape = node.in_node(0).shape
        if input_shape is None:
            return

        if not node.has_valid('spatial_dims'):
            node['spatial_dims'] = np.delete([x for x in range(len(input_shape))],
                                             [node.batch_dims[0], node.channel_dims[0]])

        input_spatial_shape = input_shape[node.spatial_dims]

        # Setting default pad and stride attrs in case of None specified
        if not node.has_valid('pad'):
            node['pad'] = int64_array([[0, 0] for x in range(len(input_shape))])
        if not node.has_valid('pad_spatial_shape'):
            node['pad_spatial_shape'] = node.pad[node.spatial_dims]
        if not node.has_valid('stride'):
            node['stride'] = int64_array([1 for x in range(len(input_shape))])

        if node.has_and_set('global_pool'):
            node['window'] = np.zeros(len(input_shape), dtype=np.int64)
            node.window[node.spatial_dims] = input_spatial_shape

        window_spatial_shape = node.window[node.spatial_dims]
        stride_spatial = node.stride[node.spatial_dims]
        assert any(stride_spatial), 'Stride can not be zero in node {}'.format(node.id)

        if node.has_valid('auto_pad') and node.auto_pad != 'explicit':
            node.pad_spatial_shape, node.output_spatial_shape = tf_window_op_pad_infer(input_spatial_shape,
                                                                                       window_spatial_shape,
                                                                                       stride_spatial, node.auto_pad)
            pad = np.zeros((len(input_shape), 2), dtype=np.int64)
            pad[node.spatial_dims] = node.pad_spatial_shape
            node.pad = pad
        else:

            pad_spatial_shape = np.add.reduce(node.pad_spatial_shape, axis=1)

            rounding = np.floor
            if node.soft_get('pooling_convention') == 'full' or node.soft_get('rounding_type') == 'ceil':
                rounding = np.ceil

            padded_spatial_shape = input_spatial_shape + pad_spatial_shape - window_spatial_shape
            if np.any(padded_spatial_shape < 0):
                raise Error("Data after padding has dimension less than window size. " +
                            "Possible reason of error is incorrectly specified model input shape(s).")

            output_spatial_shape = int64_array(rounding(float_array(padded_spatial_shape) / stride_spatial)) + 1

            original_pads = np.array([i[1] for i in node.pad_spatial_shape])

            for i in range(len(input_spatial_shape)):
                if original_pads[i] and (output_spatial_shape[i] - 1) * stride_spatial[i] >= \
                        input_spatial_shape[i] + original_pads[i]:
                    output_spatial_shape[i] -= 1

            node['output_spatial_shape'] = output_spatial_shape

        output_shape = input_shape.copy()
        output_shape[node.spatial_dims] = node.output_spatial_shape
        node.out_node().shape = output_shape

        # Add permute_attrs
        PermuteAttrs.create_permute_attrs(node, attrs=[('pad', 'input:0'),
                                                       ('stride', 'input:0'),
                                                       ('window', 'input:0'),
                                                       ('spatial_dims', 'input:0')])
