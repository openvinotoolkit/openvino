# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.ops.op import Op


class NormalizeL2Op(Op):
    op = 'NormalizeL2'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'eps': None,
            'p': None,
            'eps_mode': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer,
            'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
        }, attrs)

    def supported_attrs(self):
        return ['eps', 'eps_mode']

    @staticmethod
    def infer(node: Node):
        input_shape = node.in_port(0).data.get_shape()
        if input_shape is None:
            return

        input_value = node.in_port(0).data.get_value()
        axes = node.in_port(1).data.get_value()
        if input_value is not None and axes is not None:
            norm_value = np.linalg.norm(input_value, node.p, axes, keepdims=True)
            if node.eps_mode == 'add':
                norm_value = norm_value + node.eps
            elif node.eps_mode == 'max':
                norm_value = np.max(norm_value, node.eps)
            else:
                assert False, 'Unsupported "eps_mode" = {}'.format(node.eps_mode)
            node.out_port(0).data.set_value(input_value / norm_value)
        else:
            node.out_port(0).data.set_shape(input_shape)

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'axis')
