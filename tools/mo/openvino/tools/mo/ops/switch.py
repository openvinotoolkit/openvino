# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, is_fully_defined
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Switch(Op):
    op = 'Switch'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'infer': self.infer,
            'cf_infer': self.control_flow_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) == 2
        tensor = node.in_node(0)
        port_id = node.in_node(1)

        output_shape = shape_array(tensor.shape)
        for out_port_id in range(2):
            if node.is_out_port_connected(out_port_id):
                node.out_port(out_port_id).data.set_shape(output_shape)

        if port_id.has_valid('value'):
            output_value = tensor.value
            if output_value is not None:
                for out_port_id in range(2):
                    if node.is_out_port_connected(out_port_id):
                        node.out_port(out_port_id).data.set_value(output_value.copy())

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

        if not node_with_switch_value.has_valid('value') or not is_fully_defined(node_with_switch_value.value):
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
