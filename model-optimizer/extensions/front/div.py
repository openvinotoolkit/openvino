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

from extensions.ops.elementwise import Mul, Pow
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


class Div(FrontReplacementOp):
    op = "Div"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        power_of_exponent = Const(graph, {'value': np.float64(-1)}).create_node()
        reciprocal = Pow(graph, {'name': node.name + '/reciprocal_'}).create_node()
        mul = Mul(graph, {'name': node.name + '/mul_'}).create_node()

        # Connect nodes
        node.in_port(1).get_connection().set_destination(reciprocal.in_port(0))
        power_of_exponent.out_port(0).connect(reciprocal.in_port(1))
        node.in_port(0).get_connection().set_destination(mul.in_port(1))
        reciprocal.out_port(0).connect(mul.in_port(0))

        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [mul.id]
