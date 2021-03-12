"""
 Copyright (C) 2018-2021 Intel Corporation

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

import unittest

import numpy as np
import numpy.testing as npt

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph
from mo.utils.unittest.graph import valued_const_with_data, result, regular_op_with_empty_data, \
    shaped_const_with_data, connect
from extensions.back.MarkNodesWithShapeValues import MarkNodesWithShapeValues


class TestMarkDataTypeInShapeOfSubgraphs(unittest.TestCase):

    def test_run(self):
        inp_shape = (1, 3, 10, 10)

        nodes = {
            **shaped_const_with_data('input', int64_array(inp_shape)),
            **regular_op_with_empty_data('shapeof', {'type': 'ShapeOf'}),
            **regular_op_with_empty_data('cast', {'type': 'Cast', 'dst_type': np.float32}),
            **regular_op_with_empty_data('div', {'type': 'Mul'}),
            **valued_const_with_data('div_2_const', int64_array(2)),
            **regular_op_with_empty_data('interp', {'type': 'Interpolate', 'shape_calculation_model': 'scales',
                                                    'in_ports': {0: {}, 1: {}, 2: {}}}),
            **result('res'),
        }

        edges = [
            *connect('input', '0:interp'),
            *connect('input', '0:shapeof', skip_data=True),
            *connect('shapeof', '0:cast'),
            *connect('cast', '0:div'),
            *connect('div_2_const', '1:div'),
            *connect('div', '1:interp'),
            *connect('interp', 'res'),
        ]

        graph = build_graph(nodes, edges)
        interp_node = Node(graph, 'interp')
        interp_node.add_input_port(2, skip_if_exist=True)

        MarkNodesWithShapeValues().find_and_replace_pattern(graph)
