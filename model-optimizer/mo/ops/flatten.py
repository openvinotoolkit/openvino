# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph
from mo.ops.op import Op


class Flatten(Op):
    op = 'Flatten'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'type': None,

            'axis': None,
            'end_axis': np.int64(-1),
            'infer': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)


class FlattenONNX(Op):
    op = 'FlattenONNX'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'type': None,

            'axis': None,
            'infer': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
