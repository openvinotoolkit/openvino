"""
 Copyright (c) 2017-2019 Intel Corporation

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

from extensions.ops.elementwise import Add, Pow, Mul
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


class SquaredDifference(FrontReplacementOp):
    """
    Example class illustrating how to implement replacement of a single op in the front-end of the MO pipeline.
    This class replaces a single op "SquaredDifference" by a sub-graph consisting of 3 lower-level ops.
    """

    op = "SquaredDifference"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        # Create nodes
        const_neg = Const(graph, dict(value=np.array(-1), name=node.name + '/negate_const_')).create_node()
        negate = Mul(graph, {'name': node.name + '/negate_'}).create_node()
        add = Add(graph, {'name': node.name + '/add_'}).create_node()

        const = Const(graph, {'value': np.array(2)}).create_node()
        squared = Pow(graph, {'name': node.name + '/squared_'}).create_node()

        # Connect nodes
        node.in_port(0).get_connection().set_destination(add.in_port(0))
        node.in_port(1).get_connection().set_destination(negate.in_port(0))
        const_neg.out_port(0).connect(negate.in_port(1))
        negate.out_port(0).connect(add.in_port(1))
        add.out_port(0).connect(squared.in_port(0))
        const.out_port(0).connect(squared.in_port(1))

        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [squared.id]
