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

from extensions.ops.elementwise import Pow
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


class ReciprocalReplacer(FrontReplacementOp):
    op = "Reciprocal"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        const = Const(graph, dict(value=np.array(-1), name=node.name + '/reciprocal_pow_const_')).create_node()
        reciprocal = Pow(graph, {'name': node.name + '/reciprocal_pow_'}).create_node()
        node.in_port(0).get_connection().set_destination(reciprocal.in_port(0))
        const.out_port(0).connect(reciprocal.in_port(1))
        return [reciprocal.id]
