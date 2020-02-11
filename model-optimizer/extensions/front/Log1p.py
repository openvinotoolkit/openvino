"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.Log import LogOp
from extensions.ops.elementwise import Add
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.const import Const


class Log1p(FrontReplacementOp):
    """
    Log1p computes natural logarithm of (1 + x) element-wise.
    It replaces Log1p operation with Add -> Log.
    """
    op = "Log1p"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        const_dtype = np.float32
        if node.has_valid('data_type'):
            const_dtype = node.data_type
        const = Const(graph, {'value': np.array([1], dtype=const_dtype)}).create_node()
        add = Add(graph, {'name': node.name + '/Add_'}).create_node()
        log = LogOp(graph, {'name': node.name + '/Log_'}).create_node()

        # Connect nodes: input -> Add -> Log
        const.out_port(0).connect(add.in_port(0))
        node.in_port(0).get_connection().set_destination(add.in_port(1))
        add.out_port(0).connect(log.in_port(0))

        # The "explicit" version of the return value is: [(out_node.id, 0)])
        return [log.id]

