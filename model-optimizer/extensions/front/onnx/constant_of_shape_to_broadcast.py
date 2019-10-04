"""
 Copyright (c) 2019 Intel Corporation

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

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.broadcast import Broadcast
from mo.ops.const import Const


class ConstantOfShapeToBroadcast(FrontReplacementPattern):
    """
    Converts the 'ConstantOfShape' layer to 'Broadcast'.

    The 'ConstantOfShape' has one 1D input defining the output constant shape. The value to be filled is defined by the
    'value' attribute. The transformation creates constant node with value equal to 'value' attribute and connects it to
    the first input of a newly created 'Broadcast' node which defines value to broadcast. Then the input of the
    'ConstantOfShape' is connected to the second input of the 'Broadcast'.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for const_of_shape_node in graph.get_op_nodes(op='ConstantOfShape'):
            broadcast_node = Broadcast(graph, {'name': const_of_shape_node.name + '/Broadcast'}).create_node()
            const_of_shape_node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
            broadcast_node.in_port(0).connect(Const(graph, {'name': broadcast_node.name + '/FillValue',
                                                            'value': const_of_shape_node.fill_value}
                                                    ).create_node().out_port(0))
            const_of_shape_node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))
