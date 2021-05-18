# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.passes.eliminate import shape_inference
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op


class IfConstCondition(MiddleReplacementPattern):
    r"""
    Extracting One of If bodies if condition is const
    """
    enabled = True
    force_shape_inference = True

    @staticmethod
    def reduce_if(if_node: Node):
        cond_val = if_node.in_port(0).get_source().data.get_value()
        sub_graph = if_node['then_graph'] if cond_val else if_node['else_graph']
        mapping_dict = {}
        internal_nodes = {}
        external_nodes = {}
        for internal_node in sub_graph.get_op_nodes():
            internal_attrs = copy.deepcopy(internal_node.attrs())
            internal_attrs['name'] = 'external/' + internal_node.name
            external_node = Op(if_node.graph).create_node(attrs=internal_attrs)
            mapping_dict[internal_node.name] = external_node.name
            internal_nodes[internal_node.name] = internal_node
            external_nodes[external_node.name] = external_node

        # Create edges

        for internal_node in internal_nodes.values():
            external_node = external_nodes[mapping_dict[internal_node.name]]
            for internal_input in internal_node.in_ports().values():
                internal_connection = internal_input.get_connection()
                internal_input_node = internal_connection.get_source().node
                internal_input_node_name = internal_input_node.name

                external_input_node = external_nodes[mapping_dict[internal_input_node_name]]
                in_port_id = internal_connection.get_destination().idx
                external_node.add_input_port(in_port_id, True)
                external_node_input = external_node.in_port(in_port_id)
                out_port_id = internal_connection.get_source().idx
                external_input_node.add_output_port(out_port_id, True)
                out_port = external_input_node.out_port(out_port_id)
                out_port.get_connection().add_destination(external_node_input)

        for external_node in external_nodes.values():
            if external_node.has('input_id'):
                input_port_id = external_node['input_id']
                input_port = if_node.in_port(input_port_id)

                input_connection = input_port.get_connection()
                output_connection = external_node.out_port(0).get_connection()
                for output_port in output_connection.get_destinations():
                    output_port.disconnect()
                    input_connection.add_destination(output_port)

            if external_node.has('output_id'):
                out_port_id = external_node['output_id']
                input_connection = external_node.in_port(0).get_connection()
                external_node.in_port(0).disconnect()
                output_connection = if_node.out_port(out_port_id).get_connection()
                for output_port in output_connection.get_destinations():
                    output_port.disconnect()
                    input_connection.add_destination(output_port)
                if_node.graph.remove_node(external_node.id)  # erasing Result node

        for in_port in if_node.in_ports().values():
            in_port.disconnect()

    def find_and_replace_pattern(self, graph: Graph):
        for if_node in graph.get_op_nodes(type='If'):
            if if_node.in_port(0).get_source().node.soft_get('type') != 'Const':
                continue
            IfConstCondition.reduce_if(if_node)
