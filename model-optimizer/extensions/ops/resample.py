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

from extensions.ops.resize_factor_utils import factor_update
from mo.front.common.layout import get_batch_dim, get_features_dim, get_height_dim, get_width_dim, shape_for_layout
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class ResampleOp(Op):
    enabled = False
    op = 'Resample'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'factor': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': None
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'antialias',
            'height',
            'width',
            'resample_type',
            'factor',
        ]

    def backend_attrs(self):
        return [
            'antialias',
            'height',
            'width',
            ('type', 'resample_type'),
            'factor'
        ]

    @staticmethod
    def resample_infer(node: Node):
        layout = node.graph.graph['layout']
        assert len(layout) == 4

        input_shape = node.in_node(0).shape
        if input_shape is None:
            return
        in_height = input_shape[get_height_dim(layout, 4)]
        in_width = input_shape[get_width_dim(layout, 4)]

        if node.has('fw') and node.fw == 'tf':
            dst_shape = node.in_node(1).value
            if dst_shape is None or len(input_shape) != 4 or len(dst_shape) != 2:
                log.error(
                    'Node {} with op {} cannot be converted to Resample layer because there is no enough info about '
                    'src/dst shapes: src_shape = {}, dst_shape = {}'.format(node.name, node.op, input_shape, dst_shape))
                node.type = None  # prevent translation to a valid IE layer
                return
            out_height = dst_shape[0]
            out_width = dst_shape[1]
            node.graph.remove_edge(node.in_node(1).id, node.id)
        else:
            if len(node.in_nodes()) == 1:
                if node.has('width') and node.has('height'):
                    out_height = node.height
                    out_width = node.width
                else:
                    out_height = node.factor * in_height
                    out_width = node.factor * in_width
            else:
                out_height = node.in_node(1).shape[get_height_dim(layout, 4)]
                out_width = node.in_node(1).shape[get_width_dim(layout, 4)]

        node.factor = factor_update(
            node.factor,
            [float(out_height) / in_height, float(out_width) / in_width],
            [in_height, in_width],
            [out_height, out_width],
            node.soft_get('name'))

        node.out_node().shape = shape_for_layout(layout,
                                                 batch=input_shape[get_batch_dim(layout, 4)],
                                                 features=input_shape[get_features_dim(layout, 4)],
                                                 height=out_height,
                                                 width=out_width)
