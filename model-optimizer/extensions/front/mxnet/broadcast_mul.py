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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import replace_node
from mo.graph.graph import Node
from mo.ops.lin_op import Mul


class BroadcastMulFrontReplacer(FrontReplacementOp):
    op = 'broadcast_mul'
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        mul_op = Mul(graph, dict(name=node.id + '/mul_', symbol_dict={'name': node.id + '/mul_'}))
        mul_node = mul_op.create_node(inputs=[node.in_node(0), node.in_node(1)])
        replace_node(node, mul_node)
        return [mul_node.id]
