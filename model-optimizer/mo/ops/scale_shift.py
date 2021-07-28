# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class ScaleShiftOp(Op):
    op = 'ScaleShift'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'infer': copy_shape_infer,
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)
