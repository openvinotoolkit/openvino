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
from mo.ops.strided_slice import StridedSlice
from mo.utils.unittest.graph import build_graph
from mo.utils.unittest.graph import valued_const_with_data, result, regular_op_with_empty_data, shaped_const_with_data, \
    connect


@generator
class TestStridedSliceInfer(unittest.TestCase):
    edges = [
        *connect('input', '0:sslice'),
        *connect('begin', '1:sslice'),
        *connect('end', '2:sslice'),
        *connect('strides', '3:sslice'),
        *connect('sslice', 'res')
    ]

    def build_test_graph(self, input, begin, begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask=[0], end=[1, 34, 30, 2]):
        nodes = {
            **shaped_const_with_data('input', int64_array(input)),
            **regular_op_with_empty_data('sslice', {'op': 'StridedSlice', 'begin_mask': begin_mask, 'end_mask': end_mask,
                                                    'shrink_axis_mask': shrink_axis_mask,'ellipsis_mask': ellipsis_mask,
                                                    'new_axis_mask': new_axis_mask}),
            **valued_const_with_data('begin', int64_array(begin)),
            **valued_const_with_data('end', int64_array(end)),
            **valued_const_with_data('strides', int64_array([1])),
            **result('res'),
        }
        return build_graph(nodes, self.edges)

    @generate(*[
        # input           begin           begin_mask   end_mask      shrink_axis_mask  new_axis_mask  ref
        # inp[0:1, :, : :] res should be [0:1, ...]
        # ([1, 35, 35, 3], [0, 0, 0, 0],   [1],          [1],          [0],              [0, 0, 0, 0], [1, 34, 30, 2]),  # error
        ([1, 35, 35, 3], [0, 0, 0, 0],   [1],          [1],          [0],              [0, 0, 0, 0], [1, 35, 35, 3]),

        # inp[0:1, :, :, :2] = inp[0:1, ..., :2] out_shape = (1, 35, 35, 2)
        # ([1, 35, 35, 3], [0, 10, 10, 0], [1],          [1, 0, 0, 1], [0],              [0, 0, 0, 0], [1, 25, 25, 2]),  # error
        ([1, 35, 35, 3], [0, 10, 10, 0], [1],          [1, 0, 0, 1], [0],              [0, 0, 0, 0], [1, 35, 35, 2]),  # error

        # inp[0:1, :, :, 0:] = inp[0:1, ...] res_shape = (1, 35, 35, 3)
        # ([1, 35, 35, 3], [0, 10, 10, 0], [1, 0, 0, 1], [1],          [0],            [0, 0, 0, 0], [1, 34, 30, 2]),  # error
        ([1, 35, 35, 3], [0, 10, 10, 0], [1, 0, 0, 1], [1],          [0],              [0, 0, 0, 0], [1, 35, 35, 3]),

        ([1, 35, 35, 3], [0, 0, 0, 0],   [1],          [1, 0, 0, 1], [0],              [0, 0, 0, 0], [1, 35, 35, 2]),
        ([1, 35, 35, 3], [0, 10, 10, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0],              [0, 0, 0, 0], [1, 35, 35, 3]),
        ([1, 35, 35, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1],                [0, 0, 0, 0], [35, 35, 3]),
        ([1, 35, 35, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0],       [0, 0, 0, 1], [35, 35, 1, 3]),
        ([1, 35, 35, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0],       [0],          [35, 3]),
        ([1, 35, 35, 3], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 0],       [0],          [3]),
    ])
    def test_slice_infer_1(self, input, begin, begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ref_res):
        graph = self.build_test_graph(input, begin, begin_mask, end_mask, shrink_axis_mask, new_axis_mask)
        node = Node(graph, 'sslice')
        StridedSlice.infer(node)
        shape = node.out_port(0).data.get_shape()

        self.assertTrue(np.array_equal(shape, int64_array(ref_res)), 'Wrong output shape detected')

    def build_test_graph2(self, input, begin, end, strides, begin_mask, end_mask, shrink_axis_mask,
                          new_axis_mask, ellipsis_mask):
        nodes = {
            **valued_const_with_data('input', int64_array(input)),
            **regular_op_with_empty_data('sslice', {'op': 'StridedSlice', 'begin_mask': begin_mask, 'end_mask': end_mask,
                                                    'shrink_axis_mask': shrink_axis_mask,'ellipsis_mask': ellipsis_mask,
                                                    'new_axis_mask': new_axis_mask}),
            **valued_const_with_data('begin', int64_array(begin)),
            **valued_const_with_data('end', int64_array(end)),
            **valued_const_with_data('strides', int64_array(strides)),
            **result('res'),
        }
        return build_graph(nodes, self.edges)

    @generate(*[
        # input           begin    end    strides begin_mask   end_mask  shrink_axis_mask  new_axis_mask  ellipsis ref
        ([1, 34, 34, 62],   [0],     [4],   [1],    [0],          [1],      [0],              [0],           [0],     [1, 34, 34, 62]),
        ([1, 34, 34, 62],   [1],     [3],   [1],    [1],          [1],      [0],              [0],           [0],     [34, 34]),
        ([1, 34, 34, 62],   [0],     [4],   [1],    [0],          [1],      [0],              [1],           [0],     [[1, 34, 34, 62]]),
        ([1, 34, 34, 62],   [1],     [4],   [1],    [1],          [1],      [1],              [0],           [0],     34),
        ([1, 34, 34, 62],   [0],     [4],   [-1],   [0],          [0],      [0],              [0],           [0],     [62, 34, 34, 1]),
        ([[1, 34, 34, 62]], [0],     [4],   [1],    [1],          [1],      [1],              [0],           [0],     [1, 34, 34, 62 ]),
        ([1, 34, 34, 62],   [0],     [-1],  [1],    [1],          [1],      [0],              [0],           [0],     [1, 34, 34]),


    ])
    def test_slice_infer_2(self, input, begin, end, strides, begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis, ref_res):
        graph = self.build_test_graph2(input, begin, end, strides, begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis)
        node = Node(graph, 'sslice')
        StridedSlice.infer(node)
        self.assertTrue(np.array_equal(node.out_port(0).data.get_value(), int64_array(ref_res)), 'Wrong output value detected')

    @generate(*[
        ((10, 10, 10, 10),
                        [0,0,0], [3,0,5], [1,1,1], [1,1,1], [1,1,1],  [0], [0],   [0],   [3, 0, 5, 10]),
        ((10, 10, 10, 10),
                         [0, 0, 0, 0], [3, 0, 5, 0], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0], [0], [0], [0], [3, 0, 5, 10]),
    ])
    def test_slice_infer_3(self, input, begin, end, strides, begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis, ref_res):
        graph = self.build_test_graph(input, begin, begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis, end)
        node = Node(graph, 'sslice')
        StridedSlice.infer(node)
        shape = node.out_port(0).data.get_shape()
        self.assertTrue(np.array_equal(shape, int64_array(ref_res)), 'Wrong output shape detected')
