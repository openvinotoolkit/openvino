# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op


class ONNXResize10(Op):
    op = 'ONNXResize10'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)
