# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph
from mo.ops.op import Op


class Rank(Op):
    op = 'Rank'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': None,
            'op': self.op,

            'output_type': np.int64,
            'infer': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)
