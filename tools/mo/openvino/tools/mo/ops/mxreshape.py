# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class MXReshape(Op):
    '''
    The is internal op which use for resolving different reshape mxnet cases
    '''
    op = 'MXReshape'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'dim': None,
            'reverse': False,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': None,
        }, attrs)
