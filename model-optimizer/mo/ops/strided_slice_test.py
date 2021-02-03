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
from generator import generator, generate

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.strided_slice import StridedSlice
from mo.utils.unittest.graph import build_graph
from mo.utils.unittest.graph import valued_const_with_data, result, regular_op_with_empty_data, shaped_const_with_data, \
    connect


class TestStridedSliceInfer(unittest.TestCase):

    def run_test(self, inp, is_shape, ref_res, begin, end, strides,
                 begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask):
        input_node = shaped_const_with_data('input', int64_array(inp)) if is_shape \
            else valued_const_with_data('input', int64_array(inp))
        nodes = {
            **input_node,
            **regular_op_with_empty_data('sslice',
                                         {'op': 'StridedSlice', 'begin_mask': begin_mask, 'end_mask': end_mask,
                                          'shrink_axis_mask': shrink_axis_mask, 'ellipsis_mask': ellipsis_mask,
                                          'new_axis_mask': new_axis_mask}),
            **valued_const_with_data('begin', int64_array(begin)),
            **valued_const_with_data('end', int64_array(end)),
            **valued_const_with_data('strides', int64_array(strides)),
            **result('res'),
        }

        edges = [
            *connect('input', '0:sslice'),
            *connect('begin', '1:sslice'),
            *connect('end', '2:sslice'),
            *connect('strides', '3:sslice'),
            *connect('sslice', 'res')
        ]

        graph = build_graph(nodes, edges)
        node = Node(graph, 'sslice')
        StridedSlice.infer(node)
        res = node.out_port(0).data.get_shape() if is_shape else node.out_port(0).data.get_value()
        self.assertTrue(np.array_equal(res, ref_res))

    def test_slice_infer_value_1(self,  # inp[:4:1] = inp
                                 inp=(1, 34, 34, 62), ref_res=(1, 34, 34, 62), is_shape=False,
                                 begin=(0,), end=(4,), strides=(1,), begin_mask=(0,), end_mask=(1,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_2(self,  # inp[1:3:1] = [34, 34]
                                 inp=(1, 34, 34, 62), ref_res=(34, 34), is_shape=False,
                                 begin=(1,), end=(3,), strides=(1,), begin_mask=(1,), end_mask=(1,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_3(self,  # inp[np.newaxis, :4:1] = [[1, 34, 34, 62]], todo: carefully check
                                 inp=(1, 34, 34, 62), ref_res=((1, 34, 34, 62),), is_shape=False,
                                 begin=(0,), end=(4,), strides=(1,), begin_mask=(0,), end_mask=(1,),
                                 shrink_axis_mask=(0,), new_axis_mask=(1,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_4(self,  # inp[:4:1] = [1, 34, 34, 62]
                                 inp=(1, 34, 34, 62), ref_res=(1, 34, 34, 62), is_shape=False,
                                 begin=(0,), end=(4,), strides=(1,), begin_mask=(0,), end_mask=(1,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_5(self,  # inp[1] = 34
                                 inp=(1, 34, 34, 62), ref_res=34, is_shape=False,
                                 begin=(1,), end=(4,), strides=(1,), begin_mask=(1,), end_mask=(1,),
                                 shrink_axis_mask=(1,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_6(self,  # inp[::-1] = [62, 34, 34, 1]
                                 inp=(1, 34, 34, 62), ref_res=(62, 34, 34, 1), is_shape=False,
                                 begin=(0,), end=(4,), strides=(-1,), begin_mask=(0,), end_mask=(0,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_7(self,  # inp[0:4:1...  todo:carefully check
                                 inp=((1, 34, 34, 62),), ref_res=(1, 34, 34, 62), is_shape=False,
                                 begin=(0,), end=(4,), strides=(1,), begin_mask=(1,), end_mask=(1,),
                                 shrink_axis_mask=(1,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_8(self,  # inp[:-1:1] = [1, 34, 34], since begin_mask is [0], begin can be of any value
                                 inp=(1, 34, 34, 62), ref_res=(1, 34, 34), is_shape=False,
                                 begin=(0,), end=(-1,), strides=(1,), begin_mask=(0,), end_mask=(1,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_1(self,  # inp[]
                                 inp=(10, 10, 10, 10), ref_res=(3, 0, 5, 10), is_shape=True,
                                 begin=(0, 0, 0), end=(3, 0, 5), strides=(1, 1, 1), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_2(self,  # inp[]
                                   inp=(10, 10, 10, 10), ref_res=(3, 0, 5, 10), is_shape=True,
                                   begin=(0, 0, 0), end=(3, 0, 5), strides=(1, 1, 1), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
                                   shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_6_(self,  # inp[1:34, 0, :, :2]
                                 inp=(1, 35, 35, 3), ref_res=(1, 35, 2), is_shape=True,
                                 begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(1,), end_mask=(1, 0, 0, 1),
                                 shrink_axis_mask=(0, 1), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_6(self,  # inp[1:34, :, :, :2] begin mask is (1,) so only one value can be specified
                                 inp=(1, 35, 35, 3), ref_res=(1, 35, 2), is_shape=True,
                                 begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(1,), end_mask=(1, 0, 0, 1),
                                 shrink_axis_mask=(0, 1), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_7(self,  # inp[:, :, :, :] since all begin and end masks are zero
                                 inp=(1, 35, 35, 3), ref_res=(1, 35, 35, 3), is_shape=True,
                                 begin=(1, 10, 10, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_8(self,  # inp[0] todo: suspicious begin
                                 inp=(1, 35, 35, 3), ref_res=(35, 35, 3), is_shape=True,
                                 begin=(0,), end=(1,), strides=(1,), begin_mask=(0,), end_mask=(0,),
                                 shrink_axis_mask=(1,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_8_(self,  # inp[0, 20] ends can be of any value
                                 inp=(1, 35, 35, 3), ref_res=(35, 3), is_shape=True,
                                 begin=(0, 20), end=(1, 9999), strides=(1, 1), begin_mask=(0,), end_mask=(0,),
                                 shrink_axis_mask=(1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_9(self,  # inp[0, 0:34, 20:22, new_axis]
                                 inp=(1, 35, 35, 3), ref_res=(34, 2, 1, 3), is_shape=True,
                                 begin=(0, 0, 20, 0), end=(1, 34, 22, 2), strides=(1, 1, 1, 1), begin_mask=(0,), end_mask=(0,),
                                 shrink_axis_mask=(1,), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_10(self,  # inp[0, :34, 20, :2]
                                  inp=(1, 35, 35, 3), ref_res=(34, 2), is_shape=True,
                                  begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0,), end_mask=(0,),
                                  shrink_axis_mask=(1, 0, 1, 0), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_11(self,  # inp[0, 0, 0], since begin_mask is [0], begin can be of any value
                                  inp=(1, 35, 35, 3), ref_res=(3,), is_shape=True,
                                  begin=(0, 0, 0), end=(1, 34, 20), strides=(1, 1, 1), begin_mask=(0,), end_mask=(0,),
                                  shrink_axis_mask=(1, 1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_12(self,  # inp[..., np.newaxis]
                                  inp=(1, 35, 35), ref_res=(1, 35, 35, 1), is_shape=True,
                                  begin=(101, 0), end=(0, 0), strides=(-1, -1), begin_mask=(0, 0), end_mask=(0, 0),
                                  shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    @unittest.skip('temporary')
    def test_slice_infer_shape_14(self,  # inp[..., np.newaxis] # todo: suspicious that when begin mask is greater does not raise Exception
                                  inp=(1, 35, 35), ref_res=(1, 35, 35, 1), is_shape=True,
                                  begin=(101, 0), end=(0, 0), strides=(-1,), begin_mask=(1, 1, 0), end_mask=(0, 0),
                                  shrink_axis_mask=(0, ), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)
