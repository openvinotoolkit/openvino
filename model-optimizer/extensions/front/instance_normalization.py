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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from extensions.ops.elementwise import Add, Mul
from extensions.ops.mvn import MVN


class InstanceNormalization(FrontReplacementOp):
    ''' Decompose InstanceNormalization to scale*MVN(x) + B

        There are should be also reshapes added for each scale and B.
    '''
    op = "InstanceNormalization"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):

        # Add new nodes
        mvn = MVN(graph, {'eps': node.epsilon, 'name': node.name + '/Ins_Norm/MVN_', }).create_node()
        mul = Mul(graph, {'axis': 1, 'name': node.name + '/Ins_Norm/mul_'}).create_node()
        add = Add(graph, {'axis': 1, 'name': node.name + '/Ins_Norm/add_'}).create_node()

        # Connect nodes
        node.in_port(0).get_connection().set_destination(mvn.in_port(0))
        node.in_port(1).get_connection().set_destination(mul.in_port(1))
        node.in_port(2).get_connection().set_destination(add.in_port(1))

        mvn.out_port(0).connect(mul.in_port(0))
        mul.out_port(0).connect(add.in_port(0))

        return [add.id]
