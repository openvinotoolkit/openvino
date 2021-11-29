# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.topk import TopK
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const


class ArgOpsToTopK(MiddleReplacementPattern):
    """
        The transformation replaces ArgMax/ArgMin with the TopK layer.
    """

    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('node', dict(op=lambda x: x in ['ArgMax', 'ArgMin'])),
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['node']
        node_name = node.soft_get('name', node.id)

        connected_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_ports) == 2:
            axis = node.in_port(1).data.get_value()
        else:
            axis = node.axis

        assert axis is not None, 'The "axis" should be defined for node "{}"'.format(node_name)
        assert node.has_and_set('output_type'), 'The data type is not set for node "{}"'.format(node_name)

        topk_mode = 'max' if node.op == 'ArgMax' else 'min'
        topk_node = TopK(graph, {'axis': axis, 'mode': topk_mode, 'sort': 'index',
                                 'remove_values_output': node.has_and_set('remove_values_output'),
                                 'index_element_type': node.output_type}).create_node()
        node.in_port(0).get_connection().set_destination(topk_node.in_port(0))
        if node.has_and_set('out_max_val'):  # in this mode the ArgMax produces tuples (max_ind, max_value)
            concat_node = Concat(graph, {'axis': 1, 'name': node.name + '/Concat'}).create_node()
            concat_node.add_input_port(0, skip_if_exist=True)
            concat_node.add_input_port(1, skip_if_exist=True)
            topk_node.out_port(0).connect(concat_node.in_port(1))  # indices
            topk_node.out_port(1).connect(concat_node.in_port(0))  # values
            if not node.out_port(0).disconnected():
                node.out_port(0).get_connection().set_source(concat_node.out_port(0))
        else:
            if not node.out_port(0).disconnected():
                node.out_port(0).get_connection().set_source(topk_node.out_port(1))

        topk_node.in_port(1).connect(Const(graph, {'name': node.soft_get('name') + '/TopK',
                                                   'value': node.top_k}).create_node().out_port(0))

        graph.remove_nodes_from([node.id, node.out_node(0).id])
