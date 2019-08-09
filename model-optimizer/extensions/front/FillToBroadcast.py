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


class FillToBroadcast(FrontReplacementPattern):
    """
    Converts the 'Fill' layer to 'Broadcast'.
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for fill_node in graph.get_op_nodes(op='Fill'):
            broadcast_node = Broadcast(graph, {}).create_node()
            fill_node.in_port(0).get_connection().set_destination(broadcast_node.in_port(1))
            fill_node.in_port(1).get_connection().set_destination(broadcast_node.in_port(0))
            fill_node.out_port(0).get_connection().set_source(broadcast_node.out_port(0))
