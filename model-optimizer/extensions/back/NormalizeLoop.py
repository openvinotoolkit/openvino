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

    def run_before(self):
        return []

    def run_after(self):
        from extensions.back.pass_separator import BackFinish
        return [BackFinish]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Loop'):
            self.update_edge_attrs(node)

    @staticmethod
    def update_edge_attrs(loop_node: Node):
        # remove inputs which are not used in the body
        record_ids_to_remove = []
        for record_id, record in enumerate(loop_node.input_port_map):
            if len(loop_node.body.get_op_nodes(internal_layer_id=record['internal_layer_id'])) == 0:
                record_ids_to_remove.append(record_id)
        for record_id_to_remove in reversed(record_ids_to_remove):
            del loop_node.input_port_map[record_id_to_remove]

        # # remove outputs which are not used is not used in the body
        # record_ids_to_remove = []
        # for record_id, record in enumerate(loop_node.output_port_map):
        #     if len(loop_node.body.get_op_nodes(internal_layer_id=record['internal_layer_id'])) == 0:
        #         record_ids_to_remove.append(record_id)
        # for record_id_to_remove in reversed(record_ids_to_remove):
        #     del loop_node.output_port_map[record_id_to_remove]

        # remove unnecessary back edges
        record_ids_to_remove = []
        for record_id, record in enumerate(loop_node.back_edges):
            #            if 'purpose' in record and record['purpose'] == 'current_iteration':
            if len(loop_node.body.get_op_nodes(internal_layer_id=record['to_layer'])) == 0:
                record_ids_to_remove.append(record_id)
        for record_id_to_remove in reversed(record_ids_to_remove):
            del loop_node.back_edges[record_id_to_remove]

        # run to remove not connected output ports
        Loop.re_numerate_output_ports(loop_node)
        # to correctly generate port_map during IR generation
        for record in loop_node.output_port_map:
            pass