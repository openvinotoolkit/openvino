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

import networkx as nx

from mo.front.common.layout import get_batch_dim, shape_for_layout
from mo.graph.graph import Node
from mo.ops.op import Op


class PSROIPoolingOp(Op):
    op = 'PSROIPooling'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': PSROIPoolingOp.psroipooling_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'spatial_scale',
            'output_dim',
            'group_size'
        ]

    @staticmethod
    def psroipooling_infer(node: Node):
        """
        Sets shape of output node according specified parameters input blobs and node
        Sets number from the first input blob, channels from the second one, height and width are specified
        Parameters
        ----------
        node
        """
        shapes = [node.in_node(i).shape for i in range(len(node.in_nodes()))]
        if any(s is None for s in shapes):
            return
        layout = node.graph.graph['layout']
        assert len(layout) == 4
        node.out_node().shape = shape_for_layout(layout,
                                                 batch=shapes[1][get_batch_dim(layout, 4)],
                                                 features=node.output_dim,
                                                 height=node.group_size,
                                                 width=node.group_size)
