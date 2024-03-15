# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.ForceStrictPrecision import ForceStrictPrecision
from openvino.tools.mo.ops.prelu import PReLU
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Graph, rename_node
from openvino.tools.mo.ops.const import Const


class LeakyReLUMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ForceStrictPrecision]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('leakyrelu', dict(kind='op', op='LeakyReLU'))],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        relu = match['leakyrelu']
        relu_name = relu.soft_get('name', relu.id)
        if not relu.has_valid('negative_slope'):
            return

        rename_node(relu, relu_name + '/to_delete')
        # Create PReLU op and reconnect input/output from LeakyReLU to PReLU
        prelu = PReLU(graph, dict(name=relu_name)).create_node()
        rename_node(prelu, relu_name)

        const = Const(graph, dict(name=relu_name + "/weights", value=mo_array([relu.negative_slope]))).create_node()

        relu.in_port(0).get_connection().set_destination(prelu.in_port(0))
        const.out_port(0).connect(prelu.in_port(1))
        relu.out_port(0).get_connection().set_source(prelu.out_port(0))
