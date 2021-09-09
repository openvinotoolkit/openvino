# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.ops.op import Op

def shape_infer(node):
    # Inputs: [max_pool_input, max_pool_output, unpool_input, shape]
    assert(len(node.in_nodes()) == 4)
    node.out_node(0).shape = node.in_node(0).shape
    node.out_node(0).shape[2] = node.in_node(3).shape[2]
    node.out_node(0).shape[3] = node.in_node(3).shape[3]

class MaxUnpool(Op):
    op = 'MaxUnpool'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset8',
            'in_ports_count': 4,
            'out_ports_count': 1,
            'infer': shape_infer
        }, attrs)
