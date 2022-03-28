# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging as log

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.front.common.partial_infer.utils import int64_array


class CorrectPaddingsForPadAfterComplex(FrontReplacementSubgraph):
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.tf.SSliceComplex import SSliceComplex
        return [SSliceComplex]

    def run_after(self):
        from openvino.tools.mo.front.tf.pad_tf_to_pad import PadTFToPad
        return [PadTFToPad]

    def pattern(self):
        return dict(
            nodes=[
                ('complex', dict(op='Complex')),
                ('pad', dict(op='Pad')),
            ],
            edges=[
                ('complex', 'pad', {'in': 0}),
            ])

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        complex_node = match['complex']
        pad_node = match['pad']
        pads_begin_node = pad_node.in_port(1).get_source().node
        pads_end_node = pad_node.in_port(2).get_source().node

        pads_begin_node_name = pads_begin_node.soft_get('name', pads_begin_node.id)
        pads_end_node_name = pads_end_node.soft_get('name', pads_end_node.id)

        concat_for_pads_begin = create_op_with_const_inputs(graph, Concat,
                                                            {1: int64_array([0])},
                                                            {
                                                                'name': pads_begin_node_name + '/additional',
                                                                'in_ports_count': 2,
                                                                'axis': 0,
                                                            })
        concat_for_pads_end = create_op_with_const_inputs(graph, Concat,
                                                          {1: int64_array([0])},
                                                          {
                                                              'name': pads_end_node_name + '/additional',
                                                              'in_ports_count': 2,
                                                              'axis': 0,
                                                          })
        pad_node.in_port(1).get_source().connect(concat_for_pads_begin.in_port(0))
        pad_node.in_port(2).get_source().connect(concat_for_pads_end.in_port(0))

        pad_node.in_port(1).get_connection().set_source(concat_for_pads_begin.out_port(0))
        pad_node.in_port(2).get_connection().set_source(concat_for_pads_end.out_port(0))
