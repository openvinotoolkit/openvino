# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph
from mo.utils.error import Error


class SetPortsPattern(FrontReplacementSubgraph):
    """
    Pass used to set ports for loaded graph for Kaldi
    """
    enabled = True

    def run_before(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_after(self):
        from extensions.load.loader import LoadFinish
        return [LoadFinish]

    def find_and_replace_pattern(self, graph: Graph):
        graph.stage = 'front'
        for node_id in graph.nodes(data=False):
            node = Node(graph, node_id)
            inputs = node.get_sorted_inputs()
            outputs = node.get_sorted_outputs()

            in_ports_count = node.in_ports_count if node.has_valid('in_ports_count') else len(inputs)
            out_ports_count = node.out_ports_count if node.has_valid('out_ports_count') else len(outputs)

            if len(outputs) > out_ports_count > 1:
                raise Error("Node {} has more children than it should: " +
                            "should be {} but there is {}".format(node_id, out_ports_count, len(outputs)))

            node['_in_ports'] = {}
            node['_out_ports'] = {}
            if in_ports_count is not None:
                for idx in range(in_ports_count):
                    node.add_input_port(idx=idx)

            if out_ports_count is not None:
                for idx in range(out_ports_count):
                    node.add_output_port(idx=idx)
            idx = 0
            for in_node_id, edge_attrs in inputs:
                graph.remove_edge(in_node_id, node_id)
                if len(Node(graph, in_node_id).out_ports()) == 0:
                    Node(graph, in_node_id).add_output_port(0)
                in_node = Node(graph, in_node_id)
                in_node.out_port(edge_attrs['out']).connect(node.in_port(idx))
                # need to keep this attribute in edge for correct .mapping file generation and
                # for generation of "names" field in IR
                in_node.out_edge(edge_attrs['out'])['fw_tensor_debug_info'] = edge_attrs['fw_tensor_debug_info']
                if idx < in_ports_count - 1:
                    idx = idx + 1

            idx = 0
            for out_node_id, edge_attrs in outputs:
                graph.remove_edge(node_id, out_node_id)
                if len(Node(graph, out_node_id).in_ports()) == 0:
                    Node(graph, out_node_id).add_input_port(0)
                node.out_port(idx).connect(Node(graph, out_node_id).in_port(edge_attrs['in']))
                # need to keep this attribute in edge for correct .mapping file generation and
                # for generation of "names" field in IR
                node.out_edge(idx)['fw_tensor_debug_info'] = edge_attrs['fw_tensor_debug_info']
                if idx < out_ports_count - 1:
                    idx = idx + 1
