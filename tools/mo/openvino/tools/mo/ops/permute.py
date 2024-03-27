# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import attr_getter
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class Permute(Op):
    op = 'Permute'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'order': None,
            'type': __class__.op,
            'op': __class__.op,
            'infer': None,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return [('order', lambda node: attr_getter(node, 'order'))]
