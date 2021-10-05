# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.squeeze import Squeeze


class ArgOpsSqueeze(FrontReplacementSubgraph):
    """
        In some frameworks ArgMax/ArgMin operation has keepdims attribute that indicates whether to stay a dimension
        along which maximum is computed or not. In case of keepdims=0 this dimension should be removed but ArgMax/ArgMin
        operation in IR format is not designed to cover this case. So we should additionally add Squeeze operation right
        after ArgMax/ArgMin for this case.
    """
    enabled = True

    def pattern(self):
        return dict(nodes=[('node', dict(op=lambda x: x in ['ArgMax', 'ArgMin'], keepdims=0))],
                    edges=[])

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['node']

        connected_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        squeeze_node = Squeeze(graph, dict()).create_node([], dict(name=node.name + '/Squeeze'))
        if len(connected_ports) == 2:
            node.in_port(1).get_source().connect(squeeze_node.in_port(1))
        else:
            axis_node = Const(graph, {'value': node.axis}).create_node()
            squeeze_node.in_port(1).connect(axis_node.out_port(0))
        node.out_port(0).get_connection().set_source(squeeze_node.out_port(0))
        node.out_port(0).connect(squeeze_node.in_port(0))
        return []
