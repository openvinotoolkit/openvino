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

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.broadcast import Broadcast
from mo.ops.const import Const


class FillToBroadcast(FrontReplacementPattern):
    """
    Converts the 'Fill' layer to 'Broadcast'.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for fill_node in graph.get_op_nodes(op='Fill'):
            name = fill_node.soft_get('name', fill_node.id)

            broadcast_node = Broadcast(graph, {'name': name + '/Broadcast'}).create_node()
            fill_node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
            fill_node.in_port(1).get_connection().set_destination(broadcast_node.in_port(0))
            fill_node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))

        for fill_node in graph.get_op_nodes(op='ConstantFill'):
            name = fill_node.soft_get('name', fill_node.id)

            assert fill_node.has_valid('fill_value')
            assert fill_node.has_and_set('input_as_shape')

            const = Const(graph, {'value': np.array(fill_node.fill_value), 'name': name + '/value'}).create_node()
            broadcast_node = Broadcast(graph, {'name': name + '/Broadcast'}).create_node()
            fill_node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
            const.out_port(0).connect(broadcast_node.in_port(0))
            fill_node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))
