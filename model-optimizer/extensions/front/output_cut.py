"""
 Copyright (C) 2018-2021 Intel Corporation

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
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.extractor import add_output_ops
from mo.graph.graph import Graph, get_edge_attribute_between_nodes, set_edge_attribute_between_nodes


class OutputCut(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True
    force_clean_up = True

    def run_after(self):
        from extensions.front.user_data_repack import UserDataRepack
        return [UserDataRepack]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        add_output_ops(graph, graph.graph['packed_outputs'], inputs=graph.graph['user_shapes'])

        # For keeping tensor names information for output nodes fake outputs are added
        # to graph during the model loading. In the following code fake outputs are removed
        # and tensor names information is moved to output->Result edge.
        for node in graph.get_op_nodes(needs_removal=True):
            fw_info = None
            in_node = None
            for in_port_idx in node.in_edges():
                node_idx = node.in_edge(in_port_idx)['in']
                if node_idx in node.in_nodes():
                    in_node = node.in_node(node_idx)
                    fw_info_value = get_edge_attribute_between_nodes(in_node, node, 'fw_tensor_debug_info')
                    if fw_info_value:
                        fw_info = fw_info_value
                        break
            graph.erase_node(node)

            if fw_info is not None and in_node is not None:
                for out_idx in in_node.out_nodes():
                    set_edge_attribute_between_nodes(in_node, in_node.out_node(out_idx),
                                                     'fw_tensor_debug_info', fw_info)
