# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class RestorePorts(FrontReplacementSubgraph):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.input_cut import InputCut
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
