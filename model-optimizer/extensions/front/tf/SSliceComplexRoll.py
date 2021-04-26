# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging as log

from extensions.ops.roll import Roll
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.front.tf.graph_utils import correct_roll_axes
from mo.graph.graph import Graph, rename_nodes


class SSliceComplexRoll(FrontReplacementSubgraph):
    """
    Some TF models contain the sub-graph
               SomeOp
                 |
    --------------------------
    |                        |
    StridedSlice          StridedSlice
    |                       |
    ------------------------
         Complex
          |
          |   shift   axes
          |       |      |
         -------------------
                 Roll

    Here SomeOp is some node with real output and with the shape [N_0, ..., N_{r - 1}, 2], and StridedSlice nodes
    have output shapes [N_0, ..., N_{r - 1}].

    But MO and Inference Engine do not support for complex tensors. Hence, we need to replace this sub-graph with

         SomeOp   shift   axes
          |       |      |
         -------------------
                 Roll

    And, in such replacement, we should correct axes of Roll, because input data are real now. Namely, if
    there are negative axes for Roll, we need subtract 1 from such axes indices.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('strided_slice_real', dict(op='StridedSlice')),
                ('strided_slice_imag', dict(op='StridedSlice')),
                ('complex', dict(op='Complex')),
                ('roll', dict(op='Roll')),
            ],
            edges=[
                ('strided_slice_real', 'complex', {'in': 0}),
                ('strided_slice_imag', 'complex', {'in': 1}),
                ('complex', 'roll', {'in': 0}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        strided_slice_real = match['strided_slice_real']
        strided_slice_imag = match['strided_slice_imag']

        real_input = strided_slice_real.in_port(0).get_source().node
        imag_input = strided_slice_imag.in_port(0).get_source().node
        if real_input.id != imag_input.id:
            log.debug('The pattern does not correspond to Roll for complex tensor. Different inputs.')
            return

        roll = match['roll']
        roll_name = roll.soft_get('name', roll.id)

        new_roll = Roll(graph, {}).create_node()

        correct_roll_axes(roll)

        roll.in_port(1).get_connection().set_destination(new_roll.in_port(1))
        roll.in_port(2).get_connection().set_destination(new_roll.in_port(2))

        strided_slice_real.in_port(0).get_connection().set_destination(new_roll.in_port(0))

        roll.out_port(0).get_connection().set_source(new_roll.out_port(0))

        rename_nodes([(roll, roll_name + '/to_be_removed'), (new_roll, roll_name)])
