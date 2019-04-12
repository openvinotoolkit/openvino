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
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.power import Power


class ZerosLikeReplacer(FrontReplacementOp):
    """
    Replace TF operation ZerosLike by multiplying input tensor by zero.
    """
    op = "ZerosLike"
    enabled = True

    def nodes_to_remove(self, graph: Graph, match: dict):
        # do not remove matched node
        return []

    def replace_op(self, graph: Graph, node: Node):
        power = Power(graph, dict(scale=0, name=node.name + '/Power/')).create_node()

        # Reconnecting inputs to this new node
        node.in_port(0).get_connection().set_destination(power.in_port(0))
        node.out_port(0).get_connection().set_source(power.out_port(0))
        return [power.id]
