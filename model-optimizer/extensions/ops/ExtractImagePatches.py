"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.layout import shape_for_layout, get_batch_dim, get_features_dim
from mo.front.common.partial_infer.utils import int64_array, tf_window_op_pad_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ExtractImagePatches(Op):
    op = "ExtractImagePatches"

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset3',
            'infer': ExtractImagePatches.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return [
            ('sizes', lambda node: ','.join(map(str, node['sizes'][node.spatial_dims]))),
            ('strides', lambda node: ','.join(map(str, node['strides'][node.spatial_dims]))),
            ('rates', lambda node: ','.join(map(str, node['rates'][node.spatial_dims]))),
            'auto_pad',
        ]

    @staticmethod
    def infer(node: Node):
        assert (len(node.in_nodes()) == 1), 'Wrong input nodes number for node {} with type ExtractImagePatches'\
            .format(node.soft_get('name', node.id))
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            return

        assert len(input_shape) == 4, 'ExtractImagePatches operation supports only 4D tensors'

        layout = node.graph.graph['layout']
        N = input_shape[get_batch_dim(layout, 4)]
        C = input_shape[get_features_dim(layout, 4)]

        if not node.has_valid('batch_dims'):
            node['batch_dims'] = int64_array([0])

        if not node.has_valid('channel_dims'):
            node['channel_dims'] = int64_array([3]) if layout == 'NHWC' else int64_array([1])

        if not node.has_valid('spatial_dims'):
            node['spatial_dims'] = int64_array([1, 2]) if layout == 'NHWC' else int64_array([2, 3])

        size_spatial = int64_array(node.sizes)[node.spatial_dims]

        input_spatial_shape = input_shape[node.spatial_dims]
        stride_spatial_shape = node.strides[node.spatial_dims]

        size_extent = node.rates[node.spatial_dims] * (size_spatial - 1) + 1

        pad_spatial_shape, output_spatial_shape = tf_window_op_pad_infer(input_spatial_shape,
                                                                         size_extent,
                                                                         stride_spatial_shape,
                                                                         node.auto_pad,
                                                                         False)

        out_shape = shape_for_layout(layout,
                                     batch=N,
                                     features=int(C * np.prod(size_spatial)),
                                     height=int(output_spatial_shape[0]),
                                     width=int(output_spatial_shape[1]))

        node.out_port(0).data.set_shape(int64_array(out_shape))
