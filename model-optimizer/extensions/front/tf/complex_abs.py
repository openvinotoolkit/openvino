"""
 Copyright (C) 2018-2020 Intel Corporation

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

import logging as log

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

        rename_nodes([(complex_abs, complex_abs_name + '/to_be_removed'), (sqrt, complex_abs_name)])
