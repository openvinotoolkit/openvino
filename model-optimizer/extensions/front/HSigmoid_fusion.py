# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.front.AttributedClampNormalizer import AttributedClampNormalizer
from extensions.ops.activation_ops import HSigmoid
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes
from mo.middle.pattern_match import check_value
from mo.utils.graph import Node


def replace_with_hsigmoid(graph: Graph, first_node: Node, last_node: Node):
    # determine the input port of first and last nodes which gets the 'input' node output
    add_input_port_idx = int(first_node.in_port(0).get_connection().get_source().node.soft_get('op') == 'Const')
    last_node_name = last_node.soft_get('name', last_node.id)

    hsigmoid = HSigmoid(graph, {}).create_node()
    hsigmoid.in_port(0).connect(first_node.in_port(add_input_port_idx).get_source())
    last_node.out_port(0).get_connection().set_source(hsigmoid.out_port(0))

    rename_nodes([(last_node, last_node_name + '/TBR'), (hsigmoid, last_node_name)])


class HSigmoidWithMinMax(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Min/Max defining the HSigmoid function:
    HSigmoid(x) = Min(Max(x + 3.0, 0), 6.0) / 6.0.
    """
    enabled = True

    def run_after(self):
        return [AttributedClampNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict()),
                ('add', dict(op='Add')),
                ('const_0', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 0.0, atol=1e-6)))),
                ('const_3', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 3.0, atol=1e-6)))),
                ('const_6', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 6.0, atol=1e-6)))),
                ('const_1_6',
                 dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 1.0 / 6.0, atol=1e-6)))),
                ('max', dict(op='Maximum')),
                ('min', dict(op='Minimum')),
                ('mul_2', dict(op='Mul')),
            ],
            edges=[
                ('input', 'add', {'out': 0}),
                ('const_3', 'add', {}),
                ('add', 'max', {}),
                ('const_0', 'max', {}),
                ('max', 'min', {}),
                ('const_6', 'min', {}),
                ('min', 'mul_2', {}),
                ('const_1_6', 'mul_2', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        replace_with_hsigmoid(graph, match['add'], match['mul_2'])
