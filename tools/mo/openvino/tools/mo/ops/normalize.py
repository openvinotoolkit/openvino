# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.front.common.partial_infer.utils import mark_input_bins
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op


class NormalizeOp(Op):
    op = 'Normalize'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'eps': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': self.infer
        }, attrs)

        if 'across_spatial' in self.attrs and isinstance(self.attrs['across_spatial'], str):
            self.attrs['across_spatial'] = int(self.attrs['across_spatial'])

        if 'channel_shared' in self.attrs and isinstance(self.attrs['channel_shared'], str):
            self.attrs['channel_shared'] = int(self.attrs['channel_shared'])

        self.attrs['across_spatial'] = bool(self.attrs['across_spatial'])
        self.attrs['channel_shared'] = bool(self.attrs['channel_shared'])

    def supported_attrs(self):
        return ['eps', 'eps_mode',
                ('across_spatial',
                 lambda node: bool(node.across_spatial) if node.has_valid('across_spatial') else None),
                ('channel_shared',
                 lambda node: bool(node.channel_shared) if node.has_valid('channel_shared') else None),
                ]

    @staticmethod
    def infer(node: Node):
        mark_input_bins(node)
        copy_shape_infer(node)
