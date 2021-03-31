# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.elementwise import Add, Mul
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const


class BinaryFakeQuantizeNormalization(FrontReplacementPattern):
    """
    FakeQuantize in binary form has exceptional meaning of 1 and 2 input nodes.
    This nodes values should be equal and express threshold to quantize tensors to two levels..
    """
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('min_in', dict()),
                ('max_in', dict()),
                ('quantize', dict(op='FakeQuantize', levels=2))],
            edges=[
                ('min_in', 'quantize', {'in': 1}),
                ('max_in', 'quantize', {'in': 2})
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        quantize = match['quantize']

        sum_node = Add(graph, dict()).create_node()
        const = Const(graph, {'value': np.array(0.5)}).create_node()
        mul_node = Mul(graph, dict()).create_node()

        mul_node.in_port(0).connect(sum_node.out_port(0))
        mul_node.in_port(1).connect(const.out_port(0))

        quantize.in_port(1).get_connection().get_source().connect(sum_node.in_port(0))
        quantize.in_port(2).get_connection().get_source().connect(sum_node.in_port(1))

        quantize.in_port(1).disconnect()
        quantize.in_port(2).disconnect()

        mul_node.out_port(0).connect(quantize.in_port(1))
        mul_node.out_port(0).connect(quantize.in_port(2))
