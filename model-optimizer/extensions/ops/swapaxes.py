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

from mo.graph.graph import Node, Graph
from mo.ops.permute import Permute


class SwapAxes(Permute):
    op = 'SwapAxis'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        attrs.update({'infer': SwapAxes.infer})
        super().__init__(graph, attrs)

    @staticmethod
    def infer(node: Node):
        node.order = list(range(node.in_node().shape.size))
        node.order[node.dim2], node.order[node.dim1] = node.order[node.dim1], node.order[node.dim2]
        Permute.infer(node)
