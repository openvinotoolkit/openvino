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
from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class ShuffleChannels(Op):
    op = 'ShuffleChannels'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset3',

            'infer': self.infer,

            'axis': 1,
            'group': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return ['group', 'axis']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        assert node.soft_get('group') is not None, 'The attribute "group" must be set for node {}'.format(node_name)
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
