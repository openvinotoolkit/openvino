"""
 Copyright (C) 2018-2020 Intel Corporation

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

#! [complex:transformation]
import logging as log

import numpy as np

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


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

