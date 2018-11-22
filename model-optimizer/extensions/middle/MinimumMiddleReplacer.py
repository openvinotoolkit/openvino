"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.eltwise import Eltwise
from mo.ops.power import Power


class MinimumMiddleReplacer(MiddleReplacementPattern):
    op = "Minimum"
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('minimum', dict(kind='op', op='Minimum'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        node = match['minimum']
        # Constant propagation case
        if node.in_node(0).value is not None and node.in_node(1).value is not None:
            return

        negate_1 = Power(graph, dict(scale=-1, name=node.name + '/negate1_'))
        negate_2 = Power(graph, dict(scale=-1, name=node.name + '/negate2_'))
        maximum = Eltwise(graph, dict(operation='max', name=node.name + '/Max_'))
        negate_output = Power(graph, dict(scale=-1, name=node.name + '/negate_out_'))

        negate_output.create_node_with_data(
            inputs=[maximum.create_node_with_data([negate_1.create_node_with_data([node.in_node(0)]),
                                                   negate_2.create_node_with_data([node.in_node(1)])])],
            data_nodes=node.out_node())
        # Delete minimum vertex
        node.graph.remove_node(node.id)
