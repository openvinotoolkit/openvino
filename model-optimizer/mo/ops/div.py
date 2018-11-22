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

import numpy as np
import networkx as nx

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node
from mo.ops.eltwise import Eltwise
from mo.ops.power import Power


class Div(FrontReplacementOp):
    op = "Div"
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        reciprocal = Power(graph, dict(scale=1, power=np.float64(-1), shift=0, name=node.name + '/reciprocal_'))
        mul = Eltwise(graph, dict(operation='mul', name=node.name + '/mul_'))

        out_node = mul.create_node([(node.in_node(0), node.in_edge(0)['out']),
                                    reciprocal.create_node([(node.in_node(1), node.in_edge(1)['out'])])
                                   ])
        # Replace edge from out port 0 of the matched node with a edge from node out_node.id with port 0.
        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [out_node.id]
