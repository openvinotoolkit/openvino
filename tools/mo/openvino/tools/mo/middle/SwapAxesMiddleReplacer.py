# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const


class SwapAxisMiddleReplacer(MiddleReplacementPattern):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', op='SwapAxis'))],
            edges=[],
        )

    def replace_pattern(self, graph: Graph, match: [str, Node]):
        swapaxis = match['op']
        assert len(swapaxis.in_ports()) == 1
        assert swapaxis.has_and_set('order')
        order = swapaxis.order

        swapaxis.add_input_port(1)
        const = Const(graph, {'value': order, 'name': swapaxis.soft_get('name', swapaxis.id) + '/Order'}).create_node()
        const.out_port(0).connect(swapaxis.in_port(1))

        Transpose.update_node_stat(swapaxis, {'need_shape_inference': True})

        del swapaxis['order']
