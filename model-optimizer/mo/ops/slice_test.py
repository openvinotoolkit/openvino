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
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph, valued_const_with_data, valued_data, regular_op_with_empty_data, \
    connect, shaped_data, shaped_const_with_data


@generator
class TestSliceOp(unittest.TestCase):
        @generate(*[
            # standard case
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], [3, 2], [0, 1], [1, 1], [[5], [3], [6]]),
            # negative bounds
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], [3, -2], [0, 1], [1, 1], [[5], [3], [6]]),
            # unusual order of axes
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], [3, -2], [1, 0], [1, 1], [[2, 3, 5]]),
            # when only input_shape is defined without values (one from buttom element is shape)
            (None, [1, 2], [4, 3], [0, 1], [1, 1], [3, 1, 6], [4, 5, 6]),
            # boundary case
            (None, [0, 2], [np.iinfo(np.int32).max, 3], [0, 1], [1, 1], [4, 1, 6], [4, 5, 6]),
            # boundary case
            (None, [np.iinfo(np.int32).min, 2], [3, 3], [0, 1], [1, 1], [3, 1, 6], [4, 5, 6]),
            # 1D input
            ([1, 3, 224, 224], [1], [2], [0], [1], [3]),
            # 1D input with negative starts
            (None, [-1], [1], [0], [-1], [2], [4]),
            # 1D input with negative ends
            (None, [1], [-1], [0], [1], [2], [4]),
            # with rounding (e.g. take from 1st to 3rd with step 4 should give shape 1 not 0)
            (None, [1], [3], [0], [4], [1], [4]),
            # with rounding and negative steps (e.g. take from 1st to 3rd with step 4 should give shape 1 not 0)
            (None, [7], [3], [0], [-7], [1], [10]),
        ])
        def test_slice_infer(self, inp_value, starts, ends, axes, steps, expected, inp_shape=None):
            if inp_value is None:
                input_node = shaped_data('data_1', int64_array(inp_shape))
            else:
                input_node = valued_data('data_1', int64_array(inp_value))

            nodes = {
                **input_node,
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
            if inp_value is not None:
                self.assertTrue(np.array_equal(slice_node.out_node().value, expected))
            else:
                self.assertTrue(np.array_equal(slice_node.out_node().shape, expected))

        # negative tests
        @generate(*[
            # starts are non-constant
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], None, [3, 2], [0, 1], [1, 1], [[5], [3], [6]]),
            # ends are non-constant
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], None, [0, 1], [1, 1], [[5], [3], [6]]),
            # axes are non-constant
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], [3, -2], None, [1, 1], [[5], [3], [6]]),
            # steps are non-constant
            ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [0, 1], [3, -2], [0, 1], None, [[5], [3], [6]]),
            # 1D input with negative starts
            (None, [1], [-1], [0], [-2], [-6], [20]),
            # case when output shape has zero elements
            (None, [1], [1], [0], [1], [0], [4])
        ])
        def test_slice_infer_negative(self, inp_value, starts, ends, axes, steps, expected, inp_shape=None):
            if inp_value is None:
                input_node = shaped_data('data_1', int64_array(inp_shape))
            else:
                input_node = valued_data('data_1', int64_array(inp_value))

            def convert_args(val, name=''):
                if val is not None:
                    return valued_const_with_data(name, int64_array(val))
                else:
                    return shaped_const_with_data(name, [0])  #fake shape

            starts = convert_args(starts, 'starts')
            ends = convert_args(ends, 'ends')
            axes = convert_args(axes, 'axes')
            steps = convert_args(steps, 'steps')

            nodes = { **input_node,
                      **regular_op_with_empty_data('slice', {'op': 'Slice'}),
                      **starts, **ends, **axes, **steps }

            graph = build_graph(nodes,
                                [('data_1', 'slice'),
                                 *connect('starts', '1:slice'),
                                 *connect('ends', '2:slice'),
                                 *connect('axes', '3:slice'),
                                 *connect('steps', '4:slice'),
                                 *connect('slice', 'slice_d')])

            graph.stage = 'middle'
            slice_node = Node(graph, 'slice')
            self.assertRaises(Error, Slice.infer, slice_node)
