# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op


class Splice(Op):
    op = 'Splice'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'const_dim': 0,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': self.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        output_shape = input_shape.copy()
        output_shape[1] = node.const_dim + (input_shape[1] - node.const_dim) * len(node.context)
        node.out_port(0).data.set_shape(output_shape)
