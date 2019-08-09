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

import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import PermuteAttrs, Op


class SwapAxis(Op):
    op = 'SwapAxis'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        node['order'] = list(range(node.in_node().shape.size))
        node.order[node.dim2], node.order[node.dim1] = node.order[node.dim1], node.order[node.dim2]

        input_shape = node.in_port(0).data.get_shape().copy()
        node.out_port(0).data.set_shape(input_shape[node.order])
        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(np.transpose(node.in_port(0).data.get_value(), axes=node.order))

        PermuteAttrs.create_permute_attrs(node, attrs=[('order', 'input:0')])
