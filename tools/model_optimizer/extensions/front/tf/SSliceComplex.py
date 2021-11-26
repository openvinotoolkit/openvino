# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging as log

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph


class SSliceComplex(FrontReplacementSubgraph):
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
          |       other inputs
          |       |  ...  |
         -------------------
                 SomeOp1

    Here SomeOp is some node with real output and with the shape [N_0, ..., N_{r - 1}, 2], and StridedSlice nodes
    have output shapes [N_0, ..., N_{r - 1}].

    But MO and Inference Engine do not support complex tensors. Hence, we need to replace this sub-graph with

         SomeOp   other inputs
          |       |  ...  |
         -------------------
                 SomeOp1

    After this transformation we need to mark SomeOp1 operation that its input rank has changed because
    its inputs/attributes should probably be updated. Currently we have such a case for a Roll operation.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('strided_slice_real', dict(op='StridedSlice')),
                ('strided_slice_imag', dict(op='StridedSlice')),
                ('complex', dict(op='Complex')),
            ],
            edges=[
                ('strided_slice_real', 'complex', {'in': 0}),
                ('strided_slice_imag', 'complex', {'in': 1}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        strided_slice_real = match['strided_slice_real']
        strided_slice_imag = match['strided_slice_imag']

        real_input = strided_slice_real.in_port(0).get_source().node
        imag_input = strided_slice_imag.in_port(0).get_source().node
        if real_input.id != imag_input.id:
            log.debug('The pattern does not correspond to operation for complex tensor. Different inputs.')
            return

        complex_node = match['complex']
        for dst in complex_node.out_port(0).get_connection().get_destinations():
            after_complex_node = dst.node
            after_complex_node['input_rank_changed'] = True
        complex_node.out_port(0).get_connection().set_source(strided_slice_real.in_port(0).get_source())
