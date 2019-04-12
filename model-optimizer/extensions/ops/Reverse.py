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
from mo.graph.graph import Graph
from mo.ops.op import Op


class Reverse(Op):
    op = 'Reverse'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            # 'type': __class__.op, # Internal MO primitive
            'axis': None,
            'op': __class__.op,
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node):
        input_data_shape = node.in_node(0).shape
        assert input_data_shape is not None
        if not node.has_valid('axis'):
            assert 1 in node.in_nodes()
            assert node.in_node(1).has_valid('value')
            assert node.in_node(1).value.size == 1

            node['axis'] = node.in_node(1).value.item()
            node.in_port(1).disconnect()

        assert node.has_valid('axis')

        assert len(node.out_nodes()) == 1
        node.out_node().shape = input_data_shape.copy()
