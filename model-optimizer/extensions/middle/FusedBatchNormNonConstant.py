"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import networkx as nx

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.eltwise import Eltwise
from mo.ops.power import Power


class FusedBatchNormNonConstant(MiddleReplacementPattern):
    """
    Replaces FusedBatchNorm(input, beta, gamma, mean, variance) with non-constant mean and variance,
    but with constant beta and gamma to a sub-expression consisting of a combinatin of Eltwise and Power
    layers and ScaleShift.
    """

    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(kind='op', op='FusedBatchNorm'))],
            edges=[]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
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

        scale_mul = Eltwise(graph, dict(operation='mul', name=node.name + '/scale_mul_'))
        shift_add = Eltwise(graph, dict(operation='sum', name=node.name + '/shift_add_'))
        mean_add = Eltwise(graph, dict(operation='sum', name=node.name + '/mean_add_'))
        variance_mul = Eltwise(graph, dict(operation='mul', name=node.name + '/variance_mul_'))

        mean_negate = Power(graph, dict(scale=-1, name=node.name + '/mean_negate_'))
        mean_arg = mean_add.create_node_with_data([
            node.in_node(0),
            mean_negate.create_node_with_data([node.in_node(3)])])

        variance_square = Power(graph, dict(power=2, name=node.name + '/variance_square_'))
        variance_denom = Power(graph, dict(shift=node.eps, power=-0.5, name=node.name + '/variance_denom_'))
        variance_arg = variance_mul.create_node_with_data([
            mean_arg,
            variance_denom.create_node_with_data([node.in_node(4)])])

        shift_add.create_node_with_data([
            scale_mul.create_node_with_data([
                variance_arg,
                node.in_node(1)]),
            node.in_node(2)],
            data_nodes=node.out_node())

        node.graph.remove_node(node.id)
