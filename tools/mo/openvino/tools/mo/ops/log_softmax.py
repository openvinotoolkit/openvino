# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op, PermuteAttrs


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
