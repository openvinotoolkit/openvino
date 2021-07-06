# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.activation_ops import SoftPlus
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes
from mo.middle.pattern_match import check_value


class SoftplusFusion(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern for the Softplus function: Softplus(x) = ln(1 + e^x)
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('exp', dict(op='Exp')),
                ('add', dict(op='Add')),
                ('const_1', dict(op='Const', value=lambda v: check_value(v, lambda x: np.allclose(x, 1.0, atol=1e-6)))),
                ('ln', dict(op='Log')),
            ],
            edges=[
                ('exp', 'add', {}),
                ('const_1', 'add', {}),
                ('add', 'ln', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        ln = match['ln']
        exp = match['exp']

        ln_name = ln.soft_get('name', ln.id)

        softplus = SoftPlus(graph, {}).create_node()
        softplus.in_port(0).connect(exp.in_port(0).get_source())
        ln.out_port(0).get_connection().set_source(softplus.out_port(0))

        rename_nodes([(ln, ln_name + '/TBR'), (softplus, ln_name)])
