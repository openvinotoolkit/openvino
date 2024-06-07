# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender
from openvino.tools.mo.utils.ir_reader.layer_to_class import copy_graph_with_ops


class IfExtender(Extender):
    op = 'If'

    @staticmethod
    def set_input_output_id(subgraph, input_port_map, output_port_map, num_of_in_ports, num_of_out_ports):
        for node in subgraph.get_op_nodes():
            if not node.has_valid('id'):
                continue
            node_id = int(node.soft_get('id'))
            for if_input_mapping_elem in input_port_map:
                if node_id == if_input_mapping_elem['internal_layer_id']:
                    node['input_id'] = if_input_mapping_elem['external_port_id']
            for if_out_mapping_elem in output_port_map:
                if node_id == if_out_mapping_elem['internal_layer_id']:
                    # If external_point ID is counted with inputs 
                    if if_out_mapping_elem['external_port_id'] > num_of_out_ports:
                        node['output_id'] = if_out_mapping_elem['external_port_id'] - num_of_in_ports
                    # If external_point ID is counted from 0 
                    else:
                        node['output_id'] = if_out_mapping_elem['external_port_id']

    @staticmethod
    def extend(op: Node):
        assert op.has('then_graph'), 'There is no "then_body" attribute in the If op {}.'.format(op.name)
        assert op.has('else_graph'), 'There is no "else_body" attribute in the If op {}.'.format(op.name)
        # Now op.body is an IREngine, we need to replace it with IREngine.graph
        op.then_graph.graph.graph['cmd_params'] = op.graph.graph['cmd_params']
        op.then_graph.graph.graph['ir_version'] = op.graph.graph['ir_version']
        op.then_graph.graph.name = op.name + '/then_body'

        op.else_graph.graph.graph['cmd_params'] = op.graph.graph['cmd_params']
        op.else_graph.graph.graph['ir_version'] = op.graph.graph['ir_version']
        op.else_graph.graph.name = op.name + '/else_body'
        op.then_graph = copy_graph_with_ops(op.then_graph.graph)
        op.else_graph = copy_graph_with_ops(op.else_graph.graph)

        num_of_in_ports = len(op.in_ports())
        num_of_out_ports = len(op.out_ports())
        IfExtender.set_input_output_id(op.then_graph, op.then_input_port_map, op.then_output_port_map, num_of_in_ports, num_of_out_ports)
        IfExtender.set_input_output_id(op.else_graph, op.else_input_port_map, op.else_output_port_map, num_of_in_ports, num_of_out_ports)
