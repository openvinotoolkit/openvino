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

from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op


class Broadcast(Op):
    """ Broadcast tensor to a given shape with optional axis parameter

        Inputs:
            [0] - tensor to be broadcasted
            [1] - shape to be broadcast to
            [2] - optional axis paramater that which axis are allowed to be broadcasted
    """

    op = 'Broadcast'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'force_precision_in_ports': {1: 'int64' if graph.graph['cmd_params'].generate_experimental_IR_V10 else 'int32',
                                         2: 'int64' if graph.graph['cmd_params'].generate_experimental_IR_V10 else 'int32',
                                         },
            'infer': __class__.infer,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        # TODO Add necessary checks and asserts
        node.out_node().shape = node.in_node(1).value
        PermuteInputs().set_input_permutation(node.in_node(1), node, 'output:0', 'shape')
        if node.in_node(0).value is not None and node.in_node(1).value is not None:
            node.out_node().value = np.broadcast_to(node.in_node(0).value, node.in_node(1).value)
