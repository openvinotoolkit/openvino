# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node


def delete_selects_from(graph: Graph, node_idxs: list):
    for node_id in node_idxs:
        greater_equal = Node(graph, node_id)
        for port in greater_equal.out_port(0).get_destinations():
            port_node = port.node
            if port_node.soft_get('op') == 'Select':

                port_node.in_port(1).disconnect()
                port_node.in_port(0).disconnect()

                # Reconnect select input to next op
                next_op_input_port = port_node.out_port(0).get_destination()
                select_input = port_node.in_port(2).get_source()
                next_op_input_port.get_connection().set_source(select_input)
                graph.remove_node(port_node.id)
