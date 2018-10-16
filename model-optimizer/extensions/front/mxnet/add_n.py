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

import networkx as nx

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node
from mo.ops.eltwise import Eltwise


class AddN(FrontReplacementOp):
    op = "add_n"
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        out_node = node
        for n in range(len(node.in_nodes())-1):
            node_name = node.name + '/add_' + str(n)
            symbol_node = dict(
                op='elemwise_add',
                name = node_name
            )
            node_add = Eltwise(graph, dict(operation='sum', name = node_name, symbol_dict=symbol_node))
            input_node = out_node.in_node(0) if n == 0 else out_node
            out_node = node_add.create_node([input_node, node.in_node(n+1)])
        return [out_node.id]
