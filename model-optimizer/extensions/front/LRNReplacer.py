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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from extensions.ops.elementwise import Mul
from mo.ops.const import Const


class LRNReplacer(FrontReplacementOp):
    op = 'LRN'
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['op']

        if not node.has_valid('bias') or (node.has_valid('bias') and node.bias == 1):
            return

        # Calculate scale value & create Const op
        scale_value = np.array(1. / (pow(node.bias, node.beta)))
        node.alpha /= node.bias
        const_node = Const(graph, {'value': scale_value, 'shape': scale_value.shape,
                                   'name': node.name + "/Const_Mul_"}).create_node()

        # Create Mul node
        mul_node = Mul(graph, {'name': node.name + "/Mul_"}).create_node()

        # Connect nodes
        const_node.out_port(0).connect(mul_node.in_port(1))
        node.out_port(0).get_connection().set_source(mul_node.out_port(0))
        node.out_port(0).connect(mul_node.in_port(0))

        # Delete bias, if it is not deleted it will appear in IR v6
        del node['bias']
