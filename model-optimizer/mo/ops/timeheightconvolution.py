# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.ops.op import Op


class TimeHeightConvolutionComponent(Op):
    op = 'timeheightconvolutioncomponent'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'infer': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
