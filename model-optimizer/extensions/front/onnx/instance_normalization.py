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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node
from mo.ops.lin_op import Add, Mul
from extensions.ops.mvn import MVN


class InstanceNormalization(FrontReplacementOp):
    ''' Decompose InstanceNormalization to scale*MVN(x) + B

        There are should be also reshapes added for each scale and B.
    '''
    op = "InstanceNormalization"
    enabled = True

    def replace_op(self, graph: nx.MultiDiGraph, node: Node):
        prefix = node.name + '/InstanceNormalization'
        mvn = MVN(graph, dict(
            name=prefix + '/MVN',
            eps=node.epsilon
        ))
        mul = Mul(graph, dict(name=prefix + '/Mul', axis=1))
        add = Add(graph, dict(name=prefix + '/Add', axis=1))


        new_subgraph = add.create_node([
            mul.create_node([
                mvn.create_node([node.in_node(0)]),
                node.in_node(1)
            ]),
            node.in_node(2)
        ])

        return [new_subgraph.id]
