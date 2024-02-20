# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Mul, Add, Pow
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const


class FusedBatchNormNonConstant(MiddleReplacementPattern):
    """
    Replaces FusedBatchNorm(input, beta, gamma, mean, variance) with non-constant mean and variance,
    but with constant beta and gamma to a sub-expression consisting of a combinatin of Eltwise layers and ScaleShift.
    """

    enabled = True

    def run_after(self):
        from openvino.tools.mo.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(kind='op', op=lambda op: op in ['FusedBatchNorm', 'FusedBatchNormV2',
                                                            'FusedBatchNormV3']))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['op']
        if (node.data_format != b'NHWC' or
                len(node.in_nodes()) != 5 or
                node.in_node(0).value is not None or  # input
                node.in_node(1).value is None or  # scale
                node.in_node(2).value is None or  # offset
                node.in_node(3).value is not None or  # mean
                node.in_node(4).value is not None or  # variance
                node.in_node(1).value.ndim != 1 or
                node.in_node(2).value.ndim != 1):
            return

        scale_mul = Mul(graph, dict(name=node.name + '/scale_mul_'))
        shift_add = Add(graph, dict(name=node.name + '/shift_add_'))
        mean_add = Add(graph, dict(name=node.name + '/mean_add_'))
        variance_mul = Mul(graph, dict(name=node.name + '/variance_mul_'))

        neg_const = Const(graph, dict(value=mo_array(-1), name=node.name + '/mean_negate_'))
        mean_negate = Mul(graph, dict(name=node.name + '/mean_negate_'))
        mean_arg = mean_add.create_node_with_data([
            node.in_node(0),
            mean_negate.create_node_with_data([node.in_node(3),
                                               neg_const.create_node_with_data()
                                               ])])

        shift_const = Const(graph, dict(value=node.eps, name=node.name + '/variance_denom_shift_const_'))
        power_const = Const(graph, dict(value=-0.5, name=node.name + '/variance_denom_power_const_'))
        variance_denom_shift = Add(graph, dict(name=node.name + '/variance_denom_shift_'))
        variance_denom_power = Pow(graph, dict(name=node.name + '/variance_denom_power_'))
        variance_arg = variance_mul.create_node_with_data([
            mean_arg,
            variance_denom_power.create_node_with_data([
                variance_denom_shift.create_node_with_data([node.in_node(4), shift_const.create_node_with_data()]),
                power_const.create_node_with_data()]
            )])

        shift_add.create_node_with_data([
            scale_mul.create_node_with_data([
                variance_arg,
                node.in_node(1)]),
            node.in_node(2)],
            data_nodes=node.out_node())

        node.graph.remove_node(node.id)
