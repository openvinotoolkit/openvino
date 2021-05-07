# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Softmax(Op):
    op = 'SoftMax'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset1',
            'infer': Softmax.infer,
            'axis': 1,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['axis']

    @staticmethod
    def infer(node: Node):
        if node.axis < 0:
            node.axis = len(node.in_node().shape) + node.axis
        copy_shape_infer(node)
        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])


class SoftmaxONNX(Op):
    op = 'SoftMaxONNX'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'infer': None,
            'axis': 1,
            'type': None, # this operation will be replaced with a
                          # Reshape(Softmax(Flatten(x, axis), -1), x.shape) sub-graph
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
