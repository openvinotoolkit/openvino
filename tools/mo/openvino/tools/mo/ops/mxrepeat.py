# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class MXRepeat(Op):
    op = 'MXRepeat'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        assert 'axis' in attrs, 'MXRepeat operation should have `axis` parameter set during creation'
        assert 'repeats' in attrs, 'MXRepeat operation should have `repeats` parameter set during creation'

        super().__init__(graph, {
            'op': self.op,
            'type': None,

            # operation should be resolved on the front phase, partial inference is not needed
            'infer': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
