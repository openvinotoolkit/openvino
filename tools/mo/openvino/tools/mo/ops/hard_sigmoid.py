# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class HardSigmoid(Op):
    op = 'HardSigmoid'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'infer': self.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        input_node = node.in_node(0)
        data_value = node.in_port(0).data.get_value()
        alpha_value = node.in_port(1).data.get_value()
        beta_value = node.in_port(2).data.get_value()
        if data_value is not None and alpha_value is not None and beta_value is not None:
            node.out_port(0).data.set_value(np.clip(data_value * alpha_value + beta_value, 0, 1))
            return

        node.out_port(0).data.set_shape(input_node.shape.copy())
