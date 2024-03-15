# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class LstmNonLinearity(Op):
    """
    """
    op = 'LstmNonLinearity'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'use_dropout': False,
            'type': None,  # type is None because this operation should not appear in IR
            'infer': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
