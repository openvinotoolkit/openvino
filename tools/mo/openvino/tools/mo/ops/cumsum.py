# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.extractor import bool_to_str
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


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
