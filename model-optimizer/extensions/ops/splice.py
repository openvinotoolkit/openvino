# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph, Node
from mo.ops.op import Op


class Splice(Op):
    op = 'Splice'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': __class__.op,
            'const_dim': 0,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': __class__.infer,
        }
        super().__init__(graph, mandatory_props, attrs)

    @staticmethod
    def infer(node: Node):
        out_node = node.out_node()
        out_node.shape = node.in_node().shape.copy()
        out_node.shape[1] = node.const_dim + (node.in_node().shape[1] - node.const_dim) * len(node.context)
