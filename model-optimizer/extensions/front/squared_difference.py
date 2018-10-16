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
from mo.ops.power import Power


class SquaredDifference(FrontReplacementOp):
    """
    Example class illustrating how to implement replacement of a single op in the front-end of the MO pipeline.
    This class replaces a single op "SquaredDifference" by a sub-graph consisting of 3 lower-level ops.
    """

    op = "SquaredDifference"
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        negate = Power(graph, dict(scale=-1, name=node.name + '/negate_'))
        add = Eltwise(graph, dict(operation='sum', name=node.name + '/add_'))
        squared = Power(graph, dict(power=2, name=node.name + '/squared_'))

        out_node = squared.create_node([add.create_node([node.in_node(0), negate.create_node([node.in_node(1)])])])
        # Replace edge from out port 0 of the matched node with a edge from node out_node.id with port 0.
        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [out_node.id]
