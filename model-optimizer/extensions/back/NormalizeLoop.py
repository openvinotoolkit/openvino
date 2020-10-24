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
from extensions.ops.loop import Loop
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node


class NormalizeLoop(BackReplacementPattern):
    registered_ops = {}
    registered_cls = []
    force_clean_up = True

    def run_before(self):
        return []

    def run_after(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Loop'):
            self.update_edge_attrs(node)

    @staticmethod
    def remove_unused_ops_from_port_map(loop_node: Node, portmap: dict, portmap_attr: str, dir: [None, str] = None):
        record_ids_to_remove = []
        for record_id, record in enumerate(portmap):
            if len(loop_node.body.get_op_nodes(internal_layer_id=record[portmap_attr])) == 0:
                record_ids_to_remove.append(record_id)
        for record_id_to_remove in reversed(record_ids_to_remove):
            if dir in ['in', 'out']:
                port_to_remove = portmap[record_id_to_remove]['external_port_id']
                if port_to_remove != -1:
                    if dir == 'in':
                        loop_node.delete_input_port(port_to_remove)
                    elif dir == 'out':
                        loop_node.delete_output_port(port_to_remove)
            del portmap[record_id_to_remove]

    @staticmethod
    def update_edge_attrs(loop_node: Node):
        # remove inputs, outputs, back edges from the port map which are not used in the body and were removed by a
        # graph clean up, for example, in case of not used current_iteration body Parameter
        NormalizeLoop.remove_unused_ops_from_port_map(loop_node, loop_node.input_port_map, 'internal_layer_id', 'in')
        NormalizeLoop.remove_unused_ops_from_port_map(loop_node, loop_node.output_port_map, 'internal_layer_id', 'out')
        NormalizeLoop.remove_unused_ops_from_port_map(loop_node, loop_node.back_edges, 'to_layer')

        # remove not connected output ports
        Loop.re_numerate_input_ports(loop_node)
        Loop.re_numerate_output_ports(loop_node)

        # Loop.pull_constant_inputs_into_body(loop_node)
        #
        # # remove not connected output ports
        # Loop.re_numerate_input_ports(loop_node)
        # Loop.re_numerate_output_ports(loop_node)
        #
        # # remove inputs, outputs, back edges from the port map which are not used in the body and were removed by a
        # # graph clean up
        # NormalizeLoop.remove_unused_ops_from_port_map(loop_node, loop_node.input_port_map, 'internal_layer_id')
        # NormalizeLoop.remove_unused_ops_from_port_map(loop_node, loop_node.output_port_map, 'internal_layer_id')
        # NormalizeLoop.remove_unused_ops_from_port_map(loop_node, loop_node.back_edges, 'to_layer')
