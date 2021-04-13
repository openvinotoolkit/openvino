# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, rename_nodes
from extensions.ops.elementwise import Add, Pow
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.graph_utils import create_op_with_const_inputs


class ComplexAbs(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('complex', dict(op='Complex')),
                ('abs', dict(op='ComplexAbs')),
            ],
            edges=[
                ('complex', 'abs', {'in': 0}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        cmp = match['complex']
        complex_abs = match['abs']
        cmp_name = cmp.soft_get('name', cmp.id)
        complex_abs_name = complex_abs.soft_get('name', complex_abs.id)

        pow0 = create_op_with_const_inputs(graph, Pow, {1: np.float32(2.0)}, {})
        pow1 = create_op_with_const_inputs(graph, Pow, {1: np.float32(2.0)}, {})

        cmp.in_port(0).get_connection().set_destination(pow0.in_port(0))
        cmp.in_port(1).get_connection().set_destination(pow1.in_port(0))

        add = Add(graph, {}).create_node([pow0, pow1])
        sqrt = create_op_with_const_inputs(graph, Pow, {1: np.float32(0.5)}, {})
        add.out_port(0).connect(sqrt.in_port(0))

        complex_abs.out_port(0).get_connection().set_source(sqrt.out_port(0))

        rename_nodes([(complex_abs, complex_abs_name + '/to_be_removed'), (sqrt, complex_abs_name)])
