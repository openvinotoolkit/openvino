# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class ConstantOfShape(Op):
    """ Create a tensor of the shape specified in the first input with all values equal to attribute 'value'.
    The operation is converted to Broadcast operation
    """

    op = 'ConstantOfShape'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': None,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'fill_value': 0,
            'infer': None,
        }, attrs)

    def supported_attrs(self):
        return ['fill_value']
