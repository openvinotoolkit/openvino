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

import networkx as nx
import numpy as np
from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.error import Error


class Switch(Op):
    op = 'Switch'

    def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'infer': Switch.switch_infer,
            'cf_infer': Switch.switch_control_flow_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def switch_infer(node: Node):
        assert len(node.in_nodes()) == 2
        tensor = node.in_node(0)
        port_id = node.in_node(1)
        if not port_id.has_valid('value'):
            raise Error("Not supported type of switch (pred is variable)")
        output_shape = tensor.shape
        output_value = tensor.value
        for _, out_node in node.graph.out_edges(node.id):
            node.graph.node[out_node]['shape'] = np.array(output_shape)
            node.graph.node[out_node]['value'] = None if output_value is None else np.array(output_value)

    @staticmethod
    def switch_control_flow_infer(node: Node, is_executable: bool, mark_executability: callable):
        """
        Infers control flow through switch operation node. It marks output data nodes executability according to
        executability of current node and switch data value
        :param node: Node instance to infer control flow through
        :param is_executable: if current node is executable
        :param mark_executability: function to mark executability of node
        """
        graph = node.graph
        n = node.id

        in_edges_with_data = graph.in_edges(n, data=True)
        out_edges_with_data = graph.out_edges(n, data=True)
        node_with_switch_value = [u for u, _, attrs in in_edges_with_data if 'in' in attrs and attrs['in'] == 1][0]
        out_data_with_attrs = [(v, attrs) for u, v, attrs in out_edges_with_data if 'out' in attrs]
        switch_data_0_port_node_id = [v for v, attrs in out_data_with_attrs if attrs['out'] == 0]
        switch_data_1_port_node_id = [v for v, attrs in out_data_with_attrs if attrs['out'] == 1]
        assert 1 <= len(switch_data_0_port_node_id) + len(switch_data_1_port_node_id) <= 2

        switch_value = graph.node[node_with_switch_value]['value']
        if switch_value:
            # 1 port activation
            for n in switch_data_0_port_node_id:
                mark_executability(n, False)
            if is_executable:
                for n in switch_data_1_port_node_id:
                    mark_executability(n, True)
            else:
                for n in switch_data_1_port_node_id:
                    mark_executability(n, False)
        else:
            # 0 port activation
            for n in switch_data_1_port_node_id:
                mark_executability(n, False)
            if is_executable:
                for n in switch_data_0_port_node_id:
                    mark_executability(n, True)
            else:
                for n in switch_data_0_port_node_id:
                    mark_executability(n, False)
