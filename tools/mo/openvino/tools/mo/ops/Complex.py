# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class Complex(Op):
    op = 'Complex'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': Complex.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node: Node):
        real_shape = node.in_port(0).data.get_shape()
        imag_shape = node.in_port(1).data.get_shape()
        if real_shape is None or imag_shape is None:
            return

        assert np.array_equal(real_shape, imag_shape), \
            "Shapes of real and imaginary parts must be the same. Got: {} as real part shape " \
            "and {} as imaginary part shape for Node {} with op {}." \
            "".format(real_shape, imag_shape, node.soft_get("name", node.id), node.op)

        output_shape = np.ma.append(real_shape, 2)
        node.out_port(0).data.set_shape(output_shape)
