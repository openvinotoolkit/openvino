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
from mo.ops.op import Op


class Switch(Op):
    op = 'Switch'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'infer': __class__.infer,
            'cf_infer': __class__.control_flow_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 2
        tensor = node.in_node(0)
        port_id = node.in_node(1)

        output_shape = tensor.shape
        # Case with variable predicate
        if not port_id.has_valid('value'):
            # infer only shapes
            for _, out_node in node.graph.out_edges(node.id):
                node.graph.node[out_node]['shape'] = np.array(output_shape)
            return
        output_value = tensor.value
        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = np.array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else np.array(output_value)

    @staticmethod
    def control_flow_infer(node: Node, is_executable: bool, mark_executability: callable):
        """
        Infers control flow through switch operation node. It marks output data nodes executability according to
        executability of current node and switch data value
        :param node: Node instance to infer control flow through
        :param is_executable: if current node is executable
        :param mark_executability: function to mark executability of node
        """
        out_data_nodes = node.out_nodes(control_flow=True)
        node_with_switch_value = node.in_node(1)

        switch_data_0_port_node_id = [out_data_nodes[0].id] if 0 in out_data_nodes else []
        switch_data_1_port_node_id = [out_data_nodes[1].id] if 1 in out_data_nodes else []
        assert 1 <= len(switch_data_0_port_node_id) + len(switch_data_1_port_node_id) <= 2

        if not node_with_switch_value.has_valid('value'):
            # Mark both ports as executable
            resulting_switch_data_node_ids = switch_data_0_port_node_id + switch_data_1_port_node_id
            for n in resulting_switch_data_node_ids:
                mark_executability(n, True)
        else:
            switch_value = node_with_switch_value.value.item(0)
            resulting_switch_data_node_ids = [switch_data_0_port_node_id, switch_data_1_port_node_id]

            for n in resulting_switch_data_node_ids[not switch_value]:
                mark_executability(n, False)
            for n in resulting_switch_data_node_ids[switch_value]:
                mark_executability(n, is_executable)
