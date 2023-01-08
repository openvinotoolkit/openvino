# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [complex:transformation]

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph


class Complex(FrontReplacementSubgraph):
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

    @staticmethod
    def replace_sub_graph(graph: Graph, match: dict):
        strided_slice_real = match['strided_slice_real']
        strided_slice_imag = match['strided_slice_imag']
        complex_node = match['complex']

        # make sure that both strided slice operations get the same data as input
        assert strided_slice_real.in_port(0).get_source() == strided_slice_imag.in_port(0).get_source()

        # identify the output port of the operation producing datat for strided slice nodes
        input_node_output_port = strided_slice_real.in_port(0).get_source()
        input_node_output_port.disconnect()

        # change the connection so now all consumers of "complex_node" get data from input node of strided slice nodes
        complex_node.out_port(0).get_connection().set_source(input_node_output_port)
#! [complex:transformation]
