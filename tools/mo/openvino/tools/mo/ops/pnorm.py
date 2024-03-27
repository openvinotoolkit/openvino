# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op


class PNormOp(Op):
    """
     PNorm operation should be replaced by operations:
     Power(P) -> Reshape(n,c*g->n,g,c)-> ReduceSum(axis=1)-> Power(1/P)
    """
    op = 'pnorm'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': self.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        shape = node.in_port(0).data.get_shape().copy()
        shape[1] = shape[1] // node.group
        node.out_port(0).data.set_shape(shape)
