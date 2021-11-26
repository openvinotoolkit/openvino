# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.elementwise import Mul, Add, Pow
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Graph
from mo.ops.const import Const


class PowerToEltwises(FrontReplacementOp):
    op = "AttributedPower"
    enabled = True
    force_clean_up = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        op = match['op']
        out_port = op.in_port(0).get_source()

        if op.soft_get('scale', 1) != 1:
            const = Const(graph, {'value': np.array(op.scale)}).create_node()
            mul = Mul(graph, {'name': op.name + '/mul_'}).create_node()
            const.out_port(0).connect(mul.in_port(1))
            out_port.connect(mul.in_port(0))
            out_port = mul.out_port(0)

        if op.soft_get('shift', 0) != 0:
            const = Const(graph, {'value': np.array(op.shift)}).create_node()
            add = Add(graph, {'name': op.name + '/add_'}).create_node()
            const.out_port(0).connect(add.in_port(1))
            out_port.connect(add.in_port(0))
            out_port = add.out_port(0)

        if op.soft_get('power', 1) != 1:
            const = Const(graph, {'value': np.array(op.power)}).create_node()
            pow = Pow(graph, {'name': op.name + '/pow_'}).create_node()
            const.out_port(0).connect(pow.in_port(1))
            out_port.connect(pow.in_port(0))
            out_port = pow.out_port(0)

        op.out_port(0).get_connection().set_source(out_port)
