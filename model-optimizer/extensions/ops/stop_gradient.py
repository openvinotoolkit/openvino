# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class StopGradientOp(Op):
    op = 'StopGradient'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'identity': True,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': StopGradientOp.shape_infer
        }, attrs)

    @staticmethod
    def shape_infer(node):
        copy_shape_infer(node)

