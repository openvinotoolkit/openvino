# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.extractor import add_output_ops
from openvino.tools.mo.graph.graph import Graph, get_edge_attribute_between_nodes, set_edge_attribute_between_nodes


class OutputCut(FrontReplacementPattern):
    enabled = True
    run_not_recursively = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.front.user_data_repack import UserDataRepack
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
            out_nodes_ids = {}
            for in_port_idx in node.in_edges():
                node_idx = node.in_edge(in_port_idx)['in']
                if node_idx in node.in_nodes():
                    in_node = node.in_node(node_idx)
                    fw_info_value = get_edge_attribute_between_nodes(in_node, node, 'fw_tensor_debug_info')
                    if fw_info_value:
                        fw_info = fw_info_value
                        break
            if fw_info is not None and in_node is not None:            
                for out_idx in in_node.out_nodes():
                    out_node = in_node.out_node(out_idx)
                    out_nodes_ids[out_idx] = out_node.id
            
            graph.erase_node(node)

            if fw_info is not None and in_node is not None:
                for out_idx in in_node.out_nodes():
                    if node.id == out_nodes_ids[out_idx]:
                        set_edge_attribute_between_nodes(in_node, in_node.out_node(out_idx),
                                                     'fw_tensor_debug_info', fw_info)
