# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op


class Assign(Op):
    op = 'Assign'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset6',
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return ['variable_id']

    @staticmethod
    def infer(node: Node):
        assert node.has_valid('variable_id'), \
            "There is no required attribute variable_id in Assign op with name " + node.id
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
