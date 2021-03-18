"""
 Copyright (C) 2018-2021 Intel Corporation

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
import numpy as np

from mo.front.extractor import bool_to_str
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


def cumsum(a, axis=None, exclusive=False, reverse=False):
    if reverse:
        a = np.flip(a, axis)
    res = np.cumsum(a, axis=axis)
    if exclusive:
        res -= a
    if reverse:
        res = np.flip(res, axis)
    return res


class CumSum(Op):
    enabled = False
    op = 'CumSum'
    version = 'opset3'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': self.version,

            'infer': self.infer,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return [('exclusive', lambda node: bool_to_str(node, 'exclusive')),
                ('reverse', lambda node: bool_to_str(node, 'reverse'))]

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None, 'Input shape is None for node "{}"'.format(node_name)
        if not node.in_port(1).disconnected():
            assert len(node.in_port(1).data.get_shape()) == 0, 'Axis is not scalar for node: {}'.format(node_name)

        node.out_port(0).data.set_shape(input_shape.copy())

        input_value = node.in_port(0).data.get_value()
        if input_value is not None:
            axis = None if node.in_port(1).disconnected() else node.in_port(1).data.get_value()
            reverse = node.reverse if node.has_valid('reverse') else False
            exclusive = node.exclusive if node.has_valid('exclusive') else False
            node.out_port(0).data.set_value(cumsum(input_value, axis=axis, reverse=reverse, exclusive=exclusive))


class MXNetCumSum(Op):
    enabled = False
    op = 'MXNetCumSum'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,

            'infer': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
