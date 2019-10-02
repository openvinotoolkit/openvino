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
from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Graph, Node
from mo.ops.op import Op
import numpy as np
import logging as log


class Power(Op):
    enabled = False
    op = 'Power'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'power': 1,
            'scale': 1,
            'shift': 0,
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        """
        List of attributes that can/should be set by a client.
        """
        return ['power', 'scale', 'shift']

    @staticmethod
    def infer(node: Node):
        input_nodes_cnt = len(node.in_nodes())
        if input_nodes_cnt > 2:
            log.error('Power layer {} must have one or two inputs (given {})'.format(node.name, len(node.in_nodes())))
            return

        # In case of two inputs we should check value of the second input (should be a scalar)
        if input_nodes_cnt == 2:
            if not node.in_node(1).has_valid('value'):
                log.error('Power layer {} do not support dynamic power value'.format(node.name, len(node.in_nodes())))
                return

            if node.in_node(1).value.ndim != 0:
                log.error('Power layer {} do not support not scalar power value'.format(node.name, len(node.in_nodes())))
                return

            node['power'] = np.array(node.in_node(1).value, dtype=np.float64)
            node.graph.remove_edge(node.in_node(1).id, node.id)

        eltwise_infer(node, lambda a: np.power(a * node.scale + node.shift, node.power))
