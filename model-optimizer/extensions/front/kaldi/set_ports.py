"""
 Copyright (c) 2019 Intel Corporation

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
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph


class SetPortsPattern(FrontReplacementSubgraph):
    """
    Pass used to set ports for loaded graph for Kaldi
    """
    enabled = True

    def run_before(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_after(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        graph.stage = 'front'
        nodes = graph.nodes(data=False).keys()
        for node_id in nodes:
            node = Node(graph, node_id)
            inputs = node.get_sorted_inputs()
            outputs = node.get_sorted_outputs()

            in_ports_count = node.in_ports_count if node.has_valid('in_ports_count') else len(inputs)
            out_ports_count = node.out_ports_count if node.has_valid('out_ports_count') else len(outputs)
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
                Node(graph, in_node_id).out_port(edge_attrs['out']).connect(node.in_port(idx))
                if idx < in_ports_count-1:
                    idx = idx + 1

            idx = 0
            for out_node_id, edge_attrs in outputs:
                graph.remove_edge(node_id, out_node_id)
                if len(Node(graph, out_node_id).in_ports()) == 0:
                    Node(graph, out_node_id).add_input_port(0)
                node.out_port(idx).connect(Node(graph, out_node_id).in_port(edge_attrs['in']))
                if idx < out_ports_count-1:
                    idx = idx + 1
