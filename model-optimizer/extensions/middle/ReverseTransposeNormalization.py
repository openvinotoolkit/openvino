# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class ReverseTransposeNormalization(MiddleReplacementPattern):
    enabled = True
    force_shape_inference = True

    def pattern(self):
        return dict(
            nodes=[('transpose', dict(type='Transpose', reverse_order=True))],
            edges=[],
        )

    def replace_pattern(self, graph: Graph, match: [str, Node]):
        node = match['transpose']
        assert len(node.in_nodes()) == 1
        order = np.arange(len(node.in_port(0).data.get_shape()))[::-1]
        const = Const(graph, {'value': order, 'name': node.soft_get('name', node.id) + '/Order'}).create_node()
        node.add_input_port(1, skip_if_exist=True)
        const.out_port(0).connect(node.in_port(1))
        node['reverse_order'] = False
