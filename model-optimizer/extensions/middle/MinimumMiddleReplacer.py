"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.ops.elementwise import Maximum, Mul
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class MinimumMiddleReplacer(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('minimum', dict(kind='op', op='Minimum'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['minimum']
        # Constant propagation case
        if node.in_node(0).value is not None and node.in_node(1).value is not None:
            return

        neg_1_const = Const(graph, dict(value=np.array(-1), name=node.name + '/negate1_const'))
        neg_2_const = Const(graph, dict(value=np.array(-1), name=node.name + '/negate2_const'))
        negate_1 = Mul(graph, dict(name=node.name + '/negate1_'))
        negate_2 = Mul(graph, dict(name=node.name + '/negate2_'))
        maximum = Maximum(graph, dict(name=node.name + '/Max_'))
        negate_output_const = Const(graph, dict(value=np.array(-1), name=node.name + '/negate_out_const_'))
        negate_output = Mul(graph, dict(scale=-1, name=node.name + '/negate_out_'))

        negate_output.create_node_with_data(
            inputs=[
                maximum.create_node_with_data(
                    [negate_1.create_node_with_data([node.in_node(0), neg_1_const.create_node_with_data()]),
                     negate_2.create_node_with_data([node.in_node(1), neg_2_const.create_node_with_data()])]),
                negate_output_const.create_node_with_data()
            ],
            data_nodes=node.out_node())
        # Delete minimum vertex
        node.graph.remove_node(node.id)
