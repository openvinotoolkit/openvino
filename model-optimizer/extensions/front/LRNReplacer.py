"""
 Copyright (c) 2017-2018 Intel Corporation

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
import networkx as nx

from mo.front.common.replacement import FrontReplacementOp
from mo.ops.lin_op import Mul
from mo.ops.const import Const


class LRNReplacer(FrontReplacementOp):
    op = 'LRN'
    enabled = True

    def replace_sub_graph(self, graph: nx.MultiDiGraph, match: dict):
        node = match['op']

        if not node.has_valid('bias') or (node.has_valid('bias') and node.bias == 1):
            return

        # Calculate scale value & create Const op
        scale_value = np.array(1. / (pow(node.bias, node.beta)))
        node.alpha /= node.bias
        const_node = Const(graph, dict(value=scale_value, shape=scale_value.shape))

        # Get all outputs for LRN layer
        out_nodes = [node for node in node.out_nodes().values()]

        # Create Mul node with inputs
        mul_node = Mul(graph, dict(name=node.id + "/Mul_"))
        mnode = mul_node.create_node(inputs=[node, const_node.create_node()])

        # Move edges from LRN to Mul node
        for out_node in out_nodes:
            edge_attrs = graph.get_edge_data(node.id, out_node.id)[0]
            graph.remove_edge(node.id, out_node.id)
            graph.add_edges_from([(mnode.id, out_node.id, edge_attrs)])
