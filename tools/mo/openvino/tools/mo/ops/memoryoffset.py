# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.op import Op


class MemoryOffset(Op):
    op = 'MemoryOffset'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': 'MemoryOffset',
            'type': None,
            'pair_name': None,
            'splitted': False,
            'has_default': False,
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        if node.has_valid('element_size'):
            # element_size should be set by Kaldi loader or MemoryOffsetAdjustment or SplitRecurrentMemoryOffset
            node.out_port(0).data.set_shape(node.element_size)
        else:
            # for TDNN blocks
            copy_shape_infer(node)
