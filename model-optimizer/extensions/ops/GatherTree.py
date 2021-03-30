# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class GatherTree(Op):
    op = 'GatherTree'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'type': __class__.op,
            'version': 'opset1',
            'infer': __class__.infer,
            'in_ports_count': 4,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return []

    @staticmethod
    def infer(node: Node):
        node.out_node().shape = node.in_node(0).shape
