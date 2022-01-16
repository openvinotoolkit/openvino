# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.op import Op


class Concat(Op):
    op = 'Concat'
    enabled = True

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'axis': 1,
            'infer': concat_infer,
            'reverse_infer': self.reverse_infer,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['axis']

    @staticmethod
    def reverse_infer(node: Node):
        assert hasattr(node, 'in_ports_count')
        assert hasattr(node, 'axis')
        out_shape = node.out_port(0).data.get_shape()

        if out_shape is None:
            return

        out_shape[node.axis] = dynamic_dimension
        for i in range(node['in_ports_count']):
            in_shape = node.in_port(i).data.get_shape()
            if in_shape is None:
                node.in_port(i).data.set_shape(out_shape)
