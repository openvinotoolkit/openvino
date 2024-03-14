# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class GroupNorm(Op):
    op = 'GroupNorm'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'infer': copy_shape_infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)


class GroupNormalization(Op):
    op = 'GroupNormalization'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'infer': copy_shape_infer,
            'version': 'opset12',

            'num_groups': None,
            'epsilon': None,

            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    def backend_attrs(self):
        return ['num_groups', 'epsilon']
