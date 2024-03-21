# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class Result(Op):
    """
    Operation that should be added after the output node of the graph. It is a marker of the graph output.
    This type of nodes is used in the dead nodes elimination pass and not dumped into the IR.
    """
    op = 'Result'

    def __init__(self, graph: Graph, attrs: dict = None):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',
            'infer': lambda x: None,
            'value': None,
            'data_type': None,
            'in_ports_count': 1,
        }, attrs)
