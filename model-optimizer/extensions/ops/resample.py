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
import networkx as nx

from mo.front.common.layout import get_height_dim, get_width_dim
from mo.graph.graph import Node
from mo.ops.op import Op
from extensions.ops.resize_factor_utils import factor_update


class ResampleOp(Op):
    op = 'Resample'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'factor': None,
            'infer': ResampleOp.resample_infer
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
        height_dim = get_height_dim(node.graph.graph['layout'])
        width_dim = get_width_dim(node.graph.graph['layout'])

        input_shape = node.in_node(0).shape
        if input_shape is None:
            return
        out_shape = input_shape.copy()
        if node.has('fw') and node.fw == 'tf':
            dst_shape = node.in_node(1).value
            if dst_shape is None or len(input_shape) != 4 or len(dst_shape) != 2:
                log.error(
                    'Node {} with op {} cannot be converted to Resample layer because there is no enough info about '
                    'src/dst shapes: src_shape = {}, dst_shape = {}'.format(node.name, node.op, input_shape, dst_shape))
                node.type = None  # prevent translation to a valid IE layer
                return
            out_shape[height_dim] = dst_shape[0]
            out_shape[width_dim] = dst_shape[1]
            node.graph.remove_edge(node.in_node(1).id, node.id)
        else:
            if len(node.in_nodes()) == 1:
                if node.has('width') and node.has('height'):
                    out_shape[height_dim] = node.height
                    out_shape[width_dim] = node.width
                else:
                    out_shape[height_dim] = node.factor * input_shape[height_dim]
                    out_shape[width_dim] = node.factor * input_shape[width_dim]
            else:
                out_shape[height_dim] = node.in_node(1).shape[height_dim]
                out_shape[width_dim] = node.in_node(1).shape[width_dim]

        real_factor = [float(out_shape[height_dim])/input_shape[height_dim], float(out_shape[width_dim])/input_shape[width_dim]]
        node.factor = factor_update(
            node.factor,
            real_factor,
            [input_shape[height_dim], input_shape[width_dim]],
            [out_shape[height_dim], out_shape[width_dim]],
            node.soft_get('name'))

        node.out_node().shape = out_shape
