# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from collections.abc import Iterable

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.strided_slice import StridedSlice
from unit_tests.utils.graph import build_graph, valued_const_with_data, result, regular_op_with_empty_data, \
    shaped_const_with_data, connect


class TestStridedSliceInfer(unittest.TestCase):

    def run_test(self, inp, is_shape, ref_res, begin, end, strides,
                 begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask):
        if is_shape:
            input_node = shaped_const_with_data('input', shape_array(inp))
        else:
            input_node = valued_const_with_data('input', shape_array(inp))

        nodes = {
            **input_node,
            **regular_op_with_empty_data('sslice',
                                         {'op': 'StridedSlice', 'begin_mask': begin_mask, 'end_mask': end_mask,
                                          'shrink_axis_mask': shrink_axis_mask, 'ellipsis_mask': ellipsis_mask,
                                          'new_axis_mask': new_axis_mask}),
            **valued_const_with_data('begin', shape_array(begin)),
            **valued_const_with_data('end', shape_array(end)),
            **valued_const_with_data('strides', shape_array(strides)),
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
        if isinstance(ref_res, Iterable):
            self.assertTrue(strict_compare_tensors(res, shape_array(ref_res)))
        else:
            self.assertEqual(res, ref_res)

    def test_slice_infer_value_1( self,  # out = inp[:4:1]
                                  inp=(1, 34, 34, 62), ref_res=(1, 34, 34, 62), is_shape=False,
                                  begin=(0,), end=(4,), strides=(1,), begin_mask=(0,), end_mask=(1,),
                                  shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
                                  ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_2(self,  # inp[1:3:1] = [34, 34]
                                 inp=(1, 34, 34, 62), ref_res=(34, 34), is_shape=False,
                                 begin=(1,), end=(3,), strides=(1,), begin_mask=(1,), end_mask=(1,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_3(self,  # inp[np.newaxis, :4:1] = [[1, 34, 34, 62]]
                                 inp=(1, 34, 34, 62), ref_res=((1, 34, 34, 62),), is_shape=False,
                                 begin=(0, 0,), end=(0, 4,), strides=(1, 1), begin_mask=(0, 0), end_mask=(1, 1),
                                 shrink_axis_mask=(0,), new_axis_mask=(1,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_4(self,  # inp[1] = 34
                                 inp=(1, 34, 34, 62), ref_res=34, is_shape=False,
                                 begin=(1,), end=(4,), strides=(1,), begin_mask=(1,), end_mask=(1,),
                                 shrink_axis_mask=(1,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_5(self,  # inp[::-1] = [62, 34, 34, 1]
                                 inp=(1, 34, 34, 62), ref_res=(62, 34, 34, 1), is_shape=False,
                                 begin=(0,), end=(4,), strides=(-1,), begin_mask=(0,), end_mask=(0,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_6(self,  # inp[0, 0:4:1]
                                 inp=((1, 34, 34, 62),), ref_res=(1, 34, 34, 62), is_shape=False,
                                 begin=(0, 0), end=(0, 4), strides=(1, 1), begin_mask=(0, 1), end_mask=(0, 1),
                                 shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_7(self,  # inp[:-1:1] = [1, 34, 34], since begin_mask is [0], begin can be of any value
                                 inp=(1, 34, 34, 62), ref_res=(1, 34, 34), is_shape=False,
                                 begin=(0,), end=(-1,), strides=(1,), begin_mask=(0,), end_mask=(1,),
                                 shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_value_8(
            self,  # inp_shape = (1, 2, 4), out = inp[..., :2, None] => out_shape = (1, 2, 2, 1)
            inp=(((0, 1, 2, 3), (4, 5, 6, 7)),), ref_res=((((0.,), (1.,)), ((4.,), (5.,))),), is_shape=False,
            begin=(0, 0, 0), end=(0, 2, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 1, 0),
            shrink_axis_mask=(0, 0, 0), new_axis_mask=(0, 0, 1), ellipsis_mask=(1, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_1(
            self,  # inp[0:3, 0:1, 0:5]
            inp=(10, 10, 10, 10), ref_res=(3, 1, 5, 10), is_shape=True,
            begin=(0, 0, 0), end=(3, 1, 5), strides=(1, 1, 1), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_2(
            self,  # inp[0:3, 0:1, 5:0:-1]
            inp=(10, 10, 10, 10), ref_res=(3, 1, 5, 10), is_shape=True,
            begin=(0, 0, 5), end=(3, 1, 0), strides=(1, 1, -1), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_3(
            self,  # inp[1:34, 0, :, :2]
            inp=(1, 35, 35, 3), ref_res=(1, 35, 2), is_shape=True,
            begin=(0, 0, 0, 0), end=(1, 34, 0, 2), strides=(1, 1, 1, 1), begin_mask=(1, 1, 0, 0), end_mask=(1, 0, 0, 1),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_4(
            self,  # inp[1:34, :, :, :2] begin mask is (1,) so only one value can be specified
            inp=(1, 35, 35, 3), ref_res=(1, 35, 2), is_shape=True,
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(1, 0, 0, ), end_mask=(1, 0, 0, 1),
            shrink_axis_mask=(0, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_5(
        self,  # inp[:, :, :, :] since all begin and end masks are zero
        inp=(1, 35, 35, 3), ref_res=(1, 35, 35, 3), is_shape=True,
        begin=(1, 10, 10, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
        shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_6(
            self,  # inp[0]
            inp=(1, 35, 35, 3), ref_res=(35, 35, 3), is_shape=True,
            begin=(0,), end=(1,), strides=(1,), begin_mask=(1,), end_mask=(0,),
            shrink_axis_mask=(1,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_7(
            self,  # inp[0, 20], ends can be of any value
            inp=(1, 35, 35, 3), ref_res=(35, 3), is_shape=True,
            begin=(0, 20), end=(1, 9999), strides=(1, 1), begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_8(
            self,  # inp[0, 0:34, 20:22, new_axis], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(34, 2, 1, 3), is_shape=True,
            begin=(0, 0, 20, 0), end=(1, 34, 22, 2), strides=(1, 1, 1, 1), begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1,), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_9(
            self,  # inp[:, 0:4, 20, new_axis], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(1, 4, 1, 3), is_shape=True,
            begin=(0, 0, 20, 0), end=(0, 4, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 1, 0, 0), end_mask=(0, 1, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_10(
            self,  # inp[:, 0:4, new_axis, 20], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(1, 4, 1, 3), is_shape=True,
            begin=(0, 0, 0, 20), end=(0, 4, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 1, 0, 0), end_mask=(0, 1, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_11(
            self,  # inp[0, :, 0:34, 20:22, new_axis], both new_axis and shrink_axis are present
            inp=(1, 3, 35, 35), ref_res=(3, 34, 2, 1), is_shape=True,
            begin=(0, 0, 0, 20, 0), end=(1, 0, 34, 22, 0), strides=(1, 1, 1, 1, 1),
            begin_mask=(1, 0, 1, 1, 1), end_mask=(1, 0, 1, 1, 1),
            shrink_axis_mask=(1,), new_axis_mask=(0, 0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_12(
            self,  # inp[0, :34, 20, :2]
            inp=(1, 35, 35, 3), ref_res=(34, 2), is_shape=True,
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0, 1, 1, 1), end_mask=(0, 1, 1, 1),
            shrink_axis_mask=(1, 0, 1, 0), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_13(
            self,  # inp[0, 0, 0], since it's shrink_axis ends can be of any value
            inp=(1, 35, 35, 3), ref_res=(3,), is_shape=True,
            begin=(0, 0, 0), end=(1, 34444, 20), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(1, 1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_14(
            self,  # inp[0, 0, 0], since begin_mask is [0], begin can be of any value
            inp=(1, 35, 35, 3), ref_res=(1, 18, 18, 3), is_shape=True,
            begin=(0, 0, 0), end=(1, 35, 35), strides=(2, 2, 2), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0, 0, 0), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    # with ellipsis
    def test_slice_infer_shape_15(
            self,  # inp[..., np.newaxis]
            inp=(1, 35, 35), ref_res=(1, 35, 35, 1), is_shape=True,
            begin=(101, 0), end=(0, 0), strides=(-1, -1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_16(
            self,  # inp_shape = (1, 720, 1080), out = inp[..., :100, None] => out_shape = (1, 720, 100, 1)
            inp=(1, 720, 1080), ref_res=(1, 720, 100, 1), is_shape=True,
            begin=(0, 0, 0), end=(0, 100, 0), strides=(1, 1, 1), begin_mask=(0, 1, 0), end_mask=(0, 1, 0),
            shrink_axis_mask=(0,), new_axis_mask=(0, 0, 1), ellipsis_mask=(1,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_17(
            self,  # inp_shape = (1, 720, 1080, 3), out = inp[..., :-1] => out_shape = (1, 720, 100, 2)
            inp=(1, 720, 1080, 3), ref_res=(1, 720, 1080, 2), is_shape=True,
            begin=(0, 0), end=(0, -1), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 1),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_18(
            self,  # inp_shape = (1, 720, 1080, 3), out = inp[..., -2] => out_shape = (1, 720, 1080)
            inp=(1, 720, 1080, 3), ref_res=(1, 720, 1080), is_shape=True,
            begin=(0, -2), end=(0, 0), strides=(1, 1), begin_mask=(0, 1), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_shape_19(
            self,  # inp_shape = (1, 720, 1080, 3), out = input[..., 0:10, 0:3] => out_shape = (1, 720, 10, 3)
            inp=(1, 720, 1080, 3), ref_res=(1, 720, 10, 3), is_shape=True,
            begin=(0, 0, 0), end=(0, 10, 3), strides=(1, 1, 1), begin_mask=(0, 1, 1), end_mask=(0, 1, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(1,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_dynamic_shape_1(
            self,  # inp[0:3, 0:1, 0:5]
            inp=(dynamic_dimension_value, 10, 10, 10), ref_res=(dynamic_dimension_value, 1, 5, 10), is_shape=True,
            begin=(0, 0, 0), end=(3, 1, 5), strides=(1, 1, 1), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_dynamic_shape_2(
            self,  # inp[0:d, 0:1, 0:5]
            inp=(10, 10, 10, 10), ref_res=(dynamic_dimension_value, 1, 5, 10), is_shape=True,
            begin=(0, 0, 0), end=(dynamic_dimension_value, 1, 5), strides=(1, 1, 1), begin_mask=(1, 1, 1),
            end_mask=(1, 1, 1), shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_slice_infer_dynamic_shape_3(
            self,  # inp[1:34, 0, :, :d]
            inp=(1, 35, 35, 3), ref_res=(1, 35, dynamic_dimension_value), is_shape=True,
            begin=(0, 0, 0, 0), end=(1, 34, 0, dynamic_dimension_value), strides=(1, 1, 1, 1), begin_mask=(1, 1, 0, 0),
            end_mask=(1, 0, 0, 1), shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_0(
            self, # inp_shape = (1, 100, 200, 3), out = inp[np.newaxis, ..., 0, :], out_shape=(1, 1, 100, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_1(
            self, # inp_shape = (1, 100, 200, 3), out = inp[..., np.newaxis, 0, :], out_shape=(1, 100, 1, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_2(
            self, # inp_shape = (1, 100, 200, 3), out = inp[0, np.newaxis, ..., :], out_shape=(1, 100, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_3(
            self, # inp_shape = (1, 100, 200, 3), out = inp[0, ..., np.newaxis, :], out_shape=(100, 200, 1, 3)
            inp=(1, 100, 200, 3), ref_res=(100, 200, 1, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_4(
            self, # inp_shape = (1, 100, 200, 3), out = inp[np.newaxis, 0, ..., :], out_shape=(1, 100, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_5(
            self, # inp_shape = (1, 100, 200, 3), out = inp[..., 0, np.newaxis, :], out_shape=(1, 100, 1, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_6(
            self, # inp_shape = (1, 100, 200, 3), out = inp[np.newaxis, ..., :, 0], out_shape=(1, 1, 100, 200)
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_7(
            self, # inp_shape = (1, 100, 200, 3), out = inp[..., np.newaxis, :, 0], out_shape=(1, 100, 1, 200)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 200),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_8(
            self, # inp_shape = (1, 100, 200, 3), out = inp[0, np.newaxis, :, ...], out_shape=(1, 100, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_9(
            self, # inp_shape = (1, 100, 200, 3), out = inp[0, ..., :, np.newaxis], out_shape=(100, 200, 3, 1)
            inp=(1, 100, 200, 3), ref_res=(100, 200, 3, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 1, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_10(
            self, # inp_shape = (1, 100, 200, 3), out = inp[np.newaxis, 0, :, ...], out_shape=(1, 100, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_11(
            self, # inp_shape = (1, 100, 200, 3), out = inp[..., 0, :, np.newaxis], out_shape=(1, 100, 3, 1)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 3, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(1, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_12(
            self, # inp_shape = (1, 100, 200, 3), out = inp[np.newaxis, :, ..., 0], out_shape=(1, 1, 100, 200)
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_13(
            self, # inp_shape = (1, 100, 200, 3), out = inp[..., :, np.newaxis, 0], out_shape=(1, 100, 200, 1)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_14(
            self, # inp_shape = (1, 100, 200, 3), out = inp[0, :, np.newaxis, ...], out_shape=(100, 1, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(100, 1, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 0, 0, 1)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_15(
            self, # inp_shape = (1, 100, 200, 3), out = inp[0, :, ..., np.newaxis], out_shape=(100, 200, 3, 1)
            inp=(1, 100, 200, 3), ref_res=(100, 200, 3, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 0, 1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_16(
            self, # inp_shape = (1, 100, 200, 3), out = inp[np.newaxis, :, 0, ...], out_shape=(1, 1, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 1, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_17(
            self, # inp_shape = (1, 100, 200, 3), out = inp[..., :, 0, np.newaxis], out_shape=(1, 100, 200, 1)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(1, 0, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_18(
            self, # inp_shape = (1, 100, 200, 3), out = inp[:, np.newaxis, ..., 0], out_shape=(1, 1, 100, 200)
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_19(
            self, # inp_shape = (1, 100, 200, 3), out = inp[:, ..., np.newaxis, 0], out_shape=(1, 100, 200, 1)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_20(
            self, # inp_shape = (1, 100, 200, 3), out = inp[:, 0, np.newaxis, ...], out_shape=(1, 1, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 1, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 0, 0, 1)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_21(
            self, # inp_shape = (1, 100, 200, 3), out = inp[:, 0, ..., np.newaxis], out_shape=(1, 200, 3, 1)
            inp=(1, 100, 200, 3), ref_res=(1, 200, 3, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 0, 1, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_22(
            self, # inp_shape = (1, 100, 200, 3), out = inp[:, np.newaxis, 0, ...], out_shape=(1, 1, 200, 3)
            inp=(1, 100, 200, 3), ref_res=(1, 1, 200, 3),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_23(
            self, # inp_shape = (1, 100, 200, 3), out = inp[:, ..., 0, np.newaxis], out_shape=(1, 100, 200, 1)
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),  is_shape=True,
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 1, 0, 0)
    ):
        self.run_test(inp, is_shape, ref_res, begin, end, strides,
                      begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    # automatically generated the whole range of 2d slices over 2d, 3d and 4d input tensors
    def test_auto_infer_strided_slice_2d_over_2d_0(self):
        """
        inp_shape = (1, 100), out = inp[:, :] => out_shape = (1, 100)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_1(self):
        """
        inp_shape = (1, 100), out = inp[:, None] => out_shape = (1, 1, 100)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_2(self):
        """
        inp_shape = (1, 100), out = inp[:, 0] => out_shape = (1,)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_3(self):
        """
        inp_shape = (1, 100), out = inp[..., :] => out_shape = (1, 100)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_4(self):
        """
        inp_shape = (1, 100), out = inp[..., None] => out_shape = (1, 100, 1)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 100, 1),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_5(self):
        """
        inp_shape = (1, 100), out = inp[..., 0] => out_shape = (1,)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_6(self):
        """
        inp_shape = (1, 100), out = inp[None, :] => out_shape = (1, 1, 100)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_7(self):
        """
        inp_shape = (1, 100), out = inp[None, None] => out_shape = (1, 1, 1, 100)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 1, 1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_8(self):
        """
        inp_shape = (1, 100), out = inp[None, 0] => out_shape = (1, 100)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_9(self):
        """
        inp_shape = (1, 100), out = inp[0, :] => out_shape = (100,)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(100,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_10(self):
        """
        inp_shape = (1, 100), out = inp[0, None] => out_shape = (1, 100)
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_2d_11(self):
        """
        inp_shape = (1, 100), out = inp[0, 0] => out_shape = ()
        """
        self.run_test(
            inp=(1, 100), is_shape=True, ref_res=(),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_0(self):
        """
        inp_shape = (1, 100, 200), out = inp[:, :] => out_shape = (1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_1(self):
        """
        inp_shape = (1, 100, 200), out = inp[:, None] => out_shape = (1, 1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_2(self):
        """
        inp_shape = (1, 100, 200), out = inp[:, 0] => out_shape = (1, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_3(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., :] => out_shape = (1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_4(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., None] => out_shape = (1, 100, 200, 1)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 200, 1),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_5(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., 0] => out_shape = (1, 100)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_6(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, :] => out_shape = (1, 1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_7(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, None] => out_shape = (1, 1, 1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 1, 1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_8(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, 0] => out_shape = (1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_9(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, :] => out_shape = (100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_10(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, None] => out_shape = (1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_3d_11(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, 0] => out_shape = (200,)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(200,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_0(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, :] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_1(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, None] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_2(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, 0] => out_shape = (1, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_3(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_4(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None] => out_shape = (1, 100, 200, 3, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3, 1),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_5(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0] => out_shape = (1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_6(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_7(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, None] => out_shape = (1, 1, 1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_8(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_9(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, :] => out_shape = (100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_10(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_auto_infer_strided_slice_2d_over_4d_11(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, 0] => out_shape = (200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    # automatically generated slices from 3d to 5d d input tensors
    # fixed number of ellipsis, newaxis and shrink_axis
    def test_auto_infer_strided_slice_3d_over_3d_0(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, ..., 0] => out_shape = (1, 1, 100)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 1, 100),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 0, 1), new_axis_mask=(1, 0, 0), ellipsis_mask=(0, 1, 0)
        )

    def test_auto_infer_strided_slice_3d_over_3d_1(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., None, 0] => out_shape = (1, 100, 1)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 1),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 0, 1), new_axis_mask=(0, 1, 0), ellipsis_mask=(1, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_3d_2(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, None, ...] => out_shape = (1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 200),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(1, 0, 0), new_axis_mask=(0, 1, 0), ellipsis_mask=(0, 0, 1)
        )

    def test_auto_infer_strided_slice_3d_over_3d_3(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, ..., None] => out_shape = (100, 200, 1)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(100, 200, 1),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(1, 0, 0), new_axis_mask=(0, 0, 1), ellipsis_mask=(0, 1, 0)
        )

    def test_auto_infer_strided_slice_3d_over_3d_4(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, 0, ...] => out_shape = (1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 200),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 1, 0), new_axis_mask=(1, 0, 0), ellipsis_mask=(0, 0, 1)
        )

    def test_auto_infer_strided_slice_3d_over_3d_5(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., 0, None] => out_shape = (1, 100, 1)
        """
        self.run_test(
            inp=(1, 100, 200), is_shape=True, ref_res=(1, 100, 1),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 1, 0), new_axis_mask=(0, 0, 1), ellipsis_mask=(1, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_4d_0(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, ..., 0, :] => out_shape = (1, 1, 100, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 100, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_4d_1(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None, 0, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_4d_2(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_3d_over_4d_3(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, ..., None, :] => out_shape = (100, 200, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(100, 200, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_4d_4(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_3d_over_4d_5(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0, None, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_5d_0(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, ..., 0, :, :] => out_shape = (1, 1, 100, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 100, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_5d_1(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., None, 0, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_5d_2(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, None, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_5d_3(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, ..., None, :, :] => out_shape = (100, 200, 1, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(100, 200, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_5d_4(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, 0, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_3d_over_5d_5(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., 0, None, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_0(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, ..., 0, :] => out_shape = (1, 1, 100, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 100, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_1(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None, 0, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_2(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_3(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, ..., None, :] => out_shape = (100, 200, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(100, 200, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_4(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_5(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0, None, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_6(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, ..., :, 0] => out_shape = (1, 1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 100, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_7(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None, :, 0] => out_shape = (1, 100, 1, 200)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 1, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_8(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None, :, ...] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_auto_infer_strided_slice_4d_over_4d_9(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, ..., :, None] => out_shape = (100, 200, 3, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(100, 200, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_10(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0, :, ...] => out_shape = (1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_auto_infer_strided_slice_4d_over_4d_11(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0, :, None] => out_shape = (1, 100, 3, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_12(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, :, ..., 0] => out_shape = (1, 1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 100, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_13(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., :, None, 0] => out_shape = (1, 100, 200, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_14(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, :, None, ...] => out_shape = (100, 1, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(100, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_auto_infer_strided_slice_4d_over_4d_15(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, :, ..., None] => out_shape = (100, 200, 3, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(100, 200, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_16(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, :, 0, ...] => out_shape = (1, 1, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_auto_infer_strided_slice_4d_over_4d_17(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., :, 0, None] => out_shape = (1, 100, 200, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_18(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, None, ..., 0] => out_shape = (1, 1, 100, 200)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 100, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_19(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, ..., None, 0] => out_shape = (1, 100, 200, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_20(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, 0, None, ...] => out_shape = (1, 1, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_auto_infer_strided_slice_4d_over_4d_21(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, 0, ..., None] => out_shape = (1, 200, 3, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 200, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_4d_22(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, None, 0, ...] => out_shape = (1, 1, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_auto_infer_strided_slice_4d_over_4d_23(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, ..., 0, None] => out_shape = (1, 100, 200, 1)
        """
        self.run_test(
            inp=(1, 100, 200, 3), is_shape=True, ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_0(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, ..., 0, :, :] => out_shape = (1, 1, 100, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 100, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_1(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., None, 0, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_2(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, None, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_3(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, ..., None, :, :] => out_shape = (100, 200, 1, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(100, 200, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_4(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, 0, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_5(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., 0, None, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_6(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, ..., :, 0, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_7(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., None, :, 0, :] => out_shape = (1, 100, 1, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 1, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_8(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, None, :, ..., :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_9(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, ..., :, None, :] => out_shape = (100, 200, 10, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(100, 200, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_10(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, 0, :, ..., :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_11(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., 0, :, None, :] => out_shape = (1, 100, 10, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_12(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, :, ..., 0, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_13(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., :, None, 0, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_14(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, :, None, ..., :] => out_shape = (100, 1, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(100, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_15(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, :, ..., None, :] => out_shape = (100, 200, 10, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(100, 200, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_16(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, :, 0, ..., :] => out_shape = (1, 1, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_17(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., :, 0, None, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_18(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, None, ..., 0, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_19(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, ..., None, 0, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_20(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, 0, None, ..., :] => out_shape = (1, 1, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_21(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, 0, ..., None, :] => out_shape = (1, 200, 10, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 200, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_22(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, None, 0, ..., :] => out_shape = (1, 1, 200, 10, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_auto_infer_strided_slice_4d_over_5d_23(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, ..., 0, None, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_test(
            inp=(1, 100, 200, 10, 3), is_shape=True, ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0), end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )
