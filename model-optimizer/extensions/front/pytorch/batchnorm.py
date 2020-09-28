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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph, Node
from mo.ops.const import Const
from mo.ops.op import Op


class BatchNorm(Op):
    op = 'BatchNorm'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'eps': None,
            'infer': copy_shape_infer,
            'in_ports_count': 5,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

class BatchNorm2d(FrontReplacementOp):
    op = 'BatchNorm2d'
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        inputs = [node.in_node(i) for i in range(5)]
        bn = BatchNorm(graph, dict(name=node.name, eps=node.module.eps)).create_node(inputs)
        return [bn.id]
