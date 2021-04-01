# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.activation_ops import Swish
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes


class SwishWithSigmoidWithoutBeta(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Sigmoid defining the Swish function: Swish(x) = x * Sigmoid(x)
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('sigmoid', dict(op='Sigmoid')),
                ('mul', dict(op='Mul')),
            ],
            edges=[
                ('sigmoid', 'mul', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        sigmoid = match['sigmoid']
        mul = match['mul']
        mul_name = mul.soft_get('name', mul.id)

        # determine the input port of Mul which gets the 'input' node output
        mul_input_port_idx = int(mul.in_port(0).get_connection().get_source().node.soft_get('op') == 'Sigmoid')

        # check that the same tensor provided as input to Mul and Sigmoid
        if mul.in_port(mul_input_port_idx).get_source() != sigmoid.in_port(0).get_source():
            return

        swish = Swish(graph, {}).create_node()
        swish.in_port(0).connect(sigmoid.in_port(0).get_source())
        mul.out_port(0).get_connection().set_source(swish.out_port(0))

        rename_nodes([(mul, mul_name + '/TBR'), (swish, mul_name)])


class SwishWithSigmoidWithBeta(FrontReplacementSubgraph):
    """
    The transformation looks for the pattern with Sigmoid defining the Swish function: Swish(x) = x * Sigmoid(x * beta)
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('sigmoid', dict(op='Sigmoid')),
                ('beta', dict()),
                ('mul_beta', dict(op='Mul')),
                ('mul', dict(op='Mul')),
            ],
            edges=[
                ('beta', 'mul_beta', {}),
                ('mul_beta', 'sigmoid', {}),
                ('sigmoid', 'mul', {}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        beta = match['beta']
        mul = match['mul']
        mul_beta = match['mul_beta']
        mul_name = mul.soft_get('name', mul.id)

        # determine the input port of Muls which get the 'input' node output
        mul_beta_input_port_idx = int(mul_beta.in_port(0).get_connection().get_source().node.id == beta.id)
        mul_input_port_idx = int(mul.in_port(0).get_connection().get_source().node.soft_get('op') == 'Sigmoid')

        # check that the same tensor provided as input to Mul and MulBeta
        if mul.in_port(mul_input_port_idx).get_source() != mul_beta.in_port(mul_beta_input_port_idx).get_source():
            return

        swish = Swish(graph, {}).create_node()
        swish.in_port(0).connect(mul_beta.in_port(mul_beta_input_port_idx).get_source())

        # connect Beta value
        swish.in_port(1).connect(mul_beta.in_port(1 - mul_beta_input_port_idx).get_source())

        mul.out_port(0).get_connection().set_source(swish.out_port(0))

        rename_nodes([(mul, mul_name + '/TBR'), (swish, mul_name)])
