# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class Identity(Op):
    op = 'Identity'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,

            'identity': True,
            'infer': self.infer,
            'reverse_infer': self.reverse_infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node):
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(node.in_port(0).data.get_value())

    @staticmethod
    def reverse_infer(node):
        out_shape = node.out_port(0).data.get_shape()
        if out_shape is not None:
            node.in_port(0).data.set_shape(out_shape)

        out_value = node.out_port(0).data.get_value()
        if out_value is not None:
            node.in_port(0).data.set_value(out_value)

class IdentityN(Op):
    op = 'IdentityN'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
        }, attrs)
