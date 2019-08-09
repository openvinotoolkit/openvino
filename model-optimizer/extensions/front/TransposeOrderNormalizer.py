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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.utils.error import Error


class TransposeOrderNormalizer(FrontReplacementSubgraph):
    """
    Transpose operation requires information about order, that is represented in original frameworks differently:
        - by layer parameter
        - by 1-port input value

    TransposeOrderNormalizer reforms Transpose operations to store axis info in 1-port input.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('transpose', dict(type='Transpose'))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['transpose']
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        if len(connected_in_ports) == 1:
            if node.has_valid('order'):
                const = Const(graph, {'value': node.order}).create_node()
                node.add_input_port(1, skip_if_exist=True)
                const.out_port(0).connect(node.in_port(1))
                del graph.node[node.id]['order']
            elif node.has('order') and node.order is None:
                assert node.has_and_set('reverse_order')
            else:
                raise Error('Can not deduce transpose `order` for {}: only one in_port and no `order` parameter.'
                            ''.format(node.op))
