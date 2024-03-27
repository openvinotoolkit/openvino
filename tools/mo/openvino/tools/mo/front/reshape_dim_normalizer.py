# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.pass_separator import FrontStart
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.utils.error import Error


class ReshapeDimNormalizer(FrontReplacementSubgraph):
    """
    Reshape operation requires information about output dimensions, that is represented in original frameworks
    differently:
        - by layer parameter
        - by 1-port input value

    This transformation reforms Reshape operations to store dim info in 1-port input.
    """
    enabled = True
    force_shape_inference = True

    def run_before(self):
        return [FrontStart]

    def run_after(self):
        from openvino.tools.mo.front.freeze_placeholder_value import FreezePlaceholderValue
        return [FreezePlaceholderValue]

    def pattern(self):
        return dict(
            nodes=[
                ('reshape', dict(kind='op', op='Reshape'))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['reshape']
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_in_ports) == 1:
            if node.has('dim'):
                const = Const(graph, {'value': node.dim}).create_node()
                node.add_input_port(1, skip_if_exist=True)
                const.out_port(0).connect(node.in_port(1))
                del node['dim']
            else:
                raise Error('The `dim` attribute for node {} is not set'.format(node.op))
