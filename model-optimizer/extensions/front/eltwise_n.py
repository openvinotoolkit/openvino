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


class EltwiseNReplacement(FrontReplacementOp):
    """
    This replacer substitutes elementwise operation with more than 2 inputs with a number of simple elementwise
    operations with 2 inputs. The replacer supports operations supported by the Eltwise layer.
    """
    op = 'EltwiseN'
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        out_node = node.in_node(0)
        operation = node.operation
        for ind in range(1, len(node.in_nodes())):
            eltwise_op = Eltwise(graph, dict(operation=operation, name=node.name + '/' + operation + '_' + str(ind)))
            out_node = eltwise_op.create_node([out_node, node.in_node(ind)])
        return [out_node.id]
