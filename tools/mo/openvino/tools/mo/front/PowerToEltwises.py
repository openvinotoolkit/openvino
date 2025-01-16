# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Mul, Add, Pow
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.const import Const


class PowerToEltwises(FrontReplacementOp):
    op = "AttributedPower"
    enabled = True
    force_clean_up = True

    def replace_sub_graph(self, graph: Graph, match: dict):
        op = match['op']
        out_port = op.in_port(0).get_source()

        if op.soft_get('scale', 1) != 1:
            const = Const(graph, {'value': mo_array(op.scale)}).create_node()
            mul = Mul(graph, {'name': op.name + '/mul_'}).create_node()
            const.out_port(0).connect(mul.in_port(1))
            mul.in_port(0).get_connection().set_source(out_port)
            out_port = mul.out_port(0)

        if op.soft_get('shift', 0) != 0:
            const = Const(graph, {'value': mo_array(op.shift)}).create_node()
            add = Add(graph, {'name': op.name + '/add_'}).create_node()
            const.out_port(0).connect(add.in_port(1))
            add.in_port(0).get_connection().set_source(out_port)
            out_port = add.out_port(0)

        if op.soft_get('power', 1) != 1:
            const = Const(graph, {'value': mo_array(op.power)}).create_node()
            pow = Pow(graph, {'name': op.name + '/pow_'}).create_node()
            const.out_port(0).connect(pow.in_port(1))
            pow.in_port(0).get_connection().set_source(out_port)
            out_port = pow.out_port(0)

        op.out_port(0).get_connection().set_source(out_port)
