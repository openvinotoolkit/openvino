# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.front.Softplus_fusion import SoftplusFusion
from extensions.ops.activation_ops import Mish
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes


class MishFusion(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Softplus defining the Mish function: Mish(x) = x * tanh(SoftPlus(x)).
    """
    enabled = True

    def run_after(self):
        return [SoftplusFusion]

    def pattern(self):
        return dict(
            nodes=[
                ('mul', dict(op='Mul')),
                ('tanh', dict(op='Tanh')),
                ('softplus', dict(op='SoftPlus')),
            ],
            edges=[
                ('softplus', 'tanh'),
                ('tanh', 'mul'),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        mul = match['mul']
        mul_name = mul.soft_get('name', mul.id)
        softplus = match['softplus']

        # determine the input port of Mul which gets the 'input' node output
        input_port_idx = int(mul.in_port(0).get_connection().get_source().node.soft_get('op') == 'Tanh')

        # check that the same tensor provided as input to Mul and SoftPlus
        if mul.in_port(input_port_idx).get_source() != softplus.in_port(0).get_source():
            return

        mish = Mish(graph, {}).create_node()
        mish.in_port(0).connect(mul.in_port(input_port_idx).get_source())
        mul.out_port(0).get_connection().set_source(mish.out_port(0))

        rename_nodes([(mul, mul_name + '/TBR'), (mish, mul_name)])
