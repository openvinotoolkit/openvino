# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer, copy_value
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class FakeOutput(Op):
    """
    This op is needed only to store the output name, it will be transformed into opset op and is doing nothing
    """
    op = 'FakeOutput'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': None,
            'version': None,

            'infer': lambda n: copy_shape_infer(n, copy_value),

            'type_infer': None,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)
