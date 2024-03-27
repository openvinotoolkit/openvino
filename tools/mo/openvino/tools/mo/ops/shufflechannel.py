# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op


class ShuffleChannels(Op):
    op = 'ShuffleChannels'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset3',

            'infer': self.infer,

            'axis': 1,
            'group': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return ['group', 'axis']

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        assert node.soft_get('group') is not None, 'The attribute "group" must be set for node {}'.format(node_name)
        node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())
