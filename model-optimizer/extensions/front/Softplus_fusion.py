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
import numpy as np

from extensions.ops.activation_ops import SoftPlus
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes


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
                ('const_1', dict(op='Const', value=lambda v: v is not None and np.allclose(v, 1.0, atol=1e-6))),
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
