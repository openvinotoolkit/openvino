# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph, Node
from mo.ops.op import Op


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
            'infer': __class__.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)


    @staticmethod
    def infer(node: Node):
        if node.has_valid('element_size'):
            # element_size should be set by Kaldi loader or by MemoryOffsetAdjustment
            node.out_port(0).data.set_shape([1, node['element_size']])
        else:
            # for TDNN blocks
            copy_shape_infer(node)
