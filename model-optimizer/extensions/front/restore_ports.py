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
from mo.graph.graph import Graph


class RestorePorts(FrontReplacementSubgraph):
    enabled = True

    def run_after(self):
        from extensions.front.input_cut import InputCut
        return [InputCut]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for node_id, attrs in graph.nodes(data=True):
            if '_in_ports' not in attrs:
                attrs['_in_ports'] = set()
            if '_out_ports' not in attrs:
                attrs['_out_ports'] = set()

        for u, v, k, d in graph.edges(data=True, keys=True):
            from_node_attrs = graph.node[u]
            to_node_attrs = graph.node[v]
            is_control_flow = 'control_flow_edge' in d and d['control_flow_edge'] is True

            in_port_id = d['in'] if not is_control_flow else 'control_flow_' + str(d['in'])
            out_port_id = d['out'] if not is_control_flow else 'control_flow_' + str(d['out'])

            to_node_attrs['_in_ports'].update({in_port_id: {'control_flow': is_control_flow}})
            from_node_attrs['_out_ports'].update({out_port_id: {'control_flow': is_control_flow}})

        graph.stage = 'front'
