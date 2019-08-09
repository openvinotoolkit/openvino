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

from mo.graph.graph import Graph
from extensions.ops.elementwise import Mul
from mo.ops.const import Const
from mo.front.common.replacement import FrontReplacementSubgraph


class SoftmaxFrontReplacementSubgraph(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('softmax', dict(type='SoftMax'))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['softmax']
        if 'temperature' in node and node['temperature'] != 1.0:
            in_node = node.in_node()
            out_nodes = [node for node in node.out_nodes().values()]
            graph.remove_edge(node.in_node().id, node.id)
            temperature = np.array([1.0 / node.temperature])
            scalar_value_op = Const(graph, dict(value=temperature, shape=temperature.shape,
                                                symbol_dict={'name': node.id + '/const'}))
            mul_op = Mul(graph, dict(name=node.id + '/mul_', symbol_dict={'name': node.id + '/mul_'}))
            mul_node = mul_op.create_node(inputs=[in_node, scalar_value_op.create_node()])
            edge_attrs = graph.get_edge_data(node.id, out_nodes[0].id)[0]
            graph.add_edges_from([(mul_node.id, node.id, edge_attrs)])

