# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class FakeConvert(Op):
    """.. warning:: FakeConvert is an experimental operation and subject to change."""
    op = 'FakeConvert'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset13',
            'is_eltwise': True,
            'infer': self.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'destination_type': 'f8e4m3',
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'destination_type',
        ]

    @staticmethod
    def infer(node: Node):
        assert len(node.in_nodes()) in [2, 3]
        assert len(node.out_nodes()) == 1
        data_input = node.in_node(0)
        node.out_port(0).data.set_shape(data_input.shape)
