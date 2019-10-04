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

from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class PNormOp(Op):
    """
     PNorm operation should be replaced by operations:
     Power(P) -> Reshape(n,c*g->n,g,c)-> ReduceSum(axis=1)-> Power(1/P)
    """
    op = 'pnorm'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': __class__.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        shape = node.in_port(0).data.get_shape().copy()
        shape[1] = shape[1] / node.group
        node.out_port(0).data.set_shape(shape)
