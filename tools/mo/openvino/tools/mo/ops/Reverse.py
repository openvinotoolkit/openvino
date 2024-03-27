# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class Reverse(Op):
    op = 'Reverse'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'axis': None,
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node):
        input_shape = node.in_port(0).data.get_shape()
        input_value = node.in_port(0).data.get_value()
        assert input_shape is not None
        if not node.has_valid('axis'):
            assert 1 in node.in_nodes()
            assert node.in_node(1).has_valid('value')
            assert node.in_node(1).value.size == 1

            node['axis'] = node.in_node(1).value.item()
            node.in_port(1).disconnect()

        assert node.has_valid('axis')

        assert len(node.out_nodes()) == 1
        if input_value is not None:
            node.out_port(0).data.set_value(np.flip(input_value, node.axis))
        else:
            node.out_port(0).data.set_shape(input_shape)
