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
from mo.graph.graph import Graph
from mo.ops.op import Op


class Identity(Op):
    op = 'Identity'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,

            'identity': True,
            'infer': self.infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node):
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(node.in_port(0).data.get_value())


class IdentityN(Op):
    op = 'IdentityN'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
        }, attrs)
