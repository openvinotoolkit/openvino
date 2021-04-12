# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.graph.graph import Graph
from mo.ops.op import Op


class Roll(Op):
    """ Roll operation that shifts elements of a tensor along specified axes.
    """
    op = 'Roll'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset7',
            'infer': copy_shape_infer,
            'in_ports_count': 3,
            'out_ports_count': 1
        }, attrs)


class AttributedRoll(Op):
    """ Roll operation that shifts elements of a tensor along specified axes.

        This operation uses the same semantics as Roll but with shift and axes specified as attributes.
    """

    op = 'AttributedRoll'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset7',
            'infer': copy_shape_infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
            'shift': None,
            'axes': None
        }, attrs)
