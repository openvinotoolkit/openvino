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


from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph, Node
from mo.ops.op import Op, PermuteAttrs


class LogSoftmax(Op):
    op = 'LogSoftmax'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset5',
            'infer': self.infer,
            'axis': 1,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['axis']

    @staticmethod
    def infer(node: Node):
        assert len([port for port in node.in_ports().values() if not port.disconnected()]) == 1,\
            'LogSoftmax node with id {} have more than one port connected'.format(node.id)
        if node.axis < 0:
            node.axis = len(node.in_port(0).data.get_shape()) + node.axis
        assert 0 <= node.axis < len(node.in_port(0).data.get_shape()),\
            'LogSoftmax node with id {} has wrong axis attribute'.format(node.id)
        copy_shape_infer(node)
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])


class LogSoftmaxONNX(Op):
    op = 'LogSoftmaxONNX'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'infer': None,
            'kind': 'op',
            'axis': 1,
            'type': None,  # the operation will be replaced with a
                           # Reshape(LogSoftmax(FlattenONNX(x, axis), 1), x.shape) sub-graph
            'op': self.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
