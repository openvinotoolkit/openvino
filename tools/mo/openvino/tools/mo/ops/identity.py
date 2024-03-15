# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
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
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),

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
