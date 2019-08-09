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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Softmax(Op):
    op = 'SoftMax'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'infer': Softmax.infer,
            'kind': 'op',
            'axis': 1,
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['axis']

    @staticmethod
    def infer(node: Node):
        if node.axis < 0:
            node.axis = len(node.in_node().shape) + node.axis
        copy_shape_infer(node)
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])

