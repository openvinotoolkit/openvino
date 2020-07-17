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
import unittest

import numpy as np
from generator import generator, generate

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.slice import Slice
from mo.utils.unittest.graph import build_graph, valued_const_with_data, valued_data, regular_op_with_empty_data, \
    connect


@generator
class TestSliceOp(unittest.TestCase):
    # todo: add this case through @generate and remove
    # def test_slice_infer_non_constant(self):
    #     # Testing non-constant path case (when value in input is None)
    #     # with multiply params
    #     graph = build_graph(nodes_attributes,
    #                         [('data_1', 'slice'),
    #                          ('starts', 'slice'),
    #                          ('ends', 'slice'),
    #                          ('axes', 'slice'),
    #                          ('slice', 'data_2')],
    #                         {'data_1': {'shape': np.array([4, 5, 6])},
    #                          'starts': {'value': np.array([1, 2])},
    #                          'ends': {'value': np.array([4, 3])},
    #                          'axes': {'value': np.array([0, 1])}})
    #
    #     slice_node = Node(graph, 'slice')
    #
    #     Slice.infer(slice_node)
    #     self.assertTrue(np.array_equal(slice_node.out_node().value, None))
    #     self.assertTrue(np.array_equal(slice_node.out_node().shape, np.array([3, 1, 6])))
    #

        @generate(*[
            # negative bounds
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], [3, -2], [0, 1], [1, 1], [[5], [3], [6]]),
            # unusual order of axes
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], [3, -2], [1, 0], [1, 1], [[2, 3, 5]]),
            # second case
            ([1, 3, 224, 224], [1], [2], [0], [1], [3])
            # # third case
            # ()
        ])
        def test_slice_infer(self, inp_value, starts, ends, axes, steps, expected, inp_shape=None):
            inp_shape = int64_array(inp_value).shape if inp_value is not None else inp_shape
            inp_value = int64_array(inp_value) if inp_value is not None else inp_value
            nodes = {
                **valued_data('data_1', inp_value),
                **regular_op_with_empty_data('slice', {'op': 'Slice'}),
                **valued_const_with_data('starts', int64_array(starts)),
                **valued_const_with_data('ends', int64_array(ends)),
                **valued_const_with_data('axes', int64_array(axes)),
                **valued_const_with_data('steps', int64_array(steps)),
            }

            graph = build_graph(nodes,
                                [('data_1', 'slice'),
                                 *connect('starts', '1:slice'),
                                 *connect('ends', '2:slice'),
                                 *connect('axes', '3:slice'),
                                 *connect('steps', '4:slice'),
                                 *connect('slice', 'slice_d')])

            graph.stage = 'middle'
            slice_node = Node(graph, 'slice')

            Slice.infer(slice_node)
            self.assertTrue(np.array_equal(slice_node.out_node().value, expected))
