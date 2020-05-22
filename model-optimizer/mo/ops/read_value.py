"""
 Copyright (C) 2020 Intel Corporation

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


class ReadValue(Op):
    op = 'ReadValue'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset3',
            'infer': self.infer,
            'type_infer': self.type_infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return ['variable_id']

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(node.in_port(0).get_data_type())

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('variable_id'), \
            "There is no required attribute variable_id in ReadValue op with name " + node.id
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
