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

import networkx as nx
import numpy as np

from mo.front.common.layout import get_width_dim, get_height_dim
from mo.front.extractor import attr_getter
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class PriorBoxClusteredOp(Op):
    op = 'PriorBoxClustered'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': PriorBoxClusteredOp.priorbox_clustered_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'width',
            'height',
            'flip',
            'clip',
            'variance',
            'img_size',
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset'
        ]

    def backend_attrs(self):
        return [
            'flip',
            'clip',
            'img_size',
            'img_h',
            'img_w',
            'step',
            'step_h',
            'step_w',
            'offset',
            ('variance', lambda node: attr_getter(node, 'variance')),
            ('width', lambda node: attr_getter(node, 'width')),
            ('height', lambda node: attr_getter(node, 'height'))
        ]

    @staticmethod
    def priorbox_clustered_infer(node: Node):
        layout = node.graph.graph['layout']
        data_shape = node.in_node(0).shape
        num_ratios = len(node.width)

        res_prod = data_shape[get_height_dim(layout, 4)] * data_shape[get_width_dim(layout, 4)] * num_ratios * 4
        node.out_node(0).shape = np.array([1, 2, res_prod], dtype=np.int64)
