# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class Roll(Op):
    op = 'Roll'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'out_ports_count': 1,
            'in_ports_count': 3,
            'infer': copy_shape_infer,
            'version': 'opset7'
        }
        super().__init__(graph, mandatory_props, attrs)
