# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import numpy.testing as npt

from openvino.tools.mo.middle.StridedSliceNormalizer import StridedSliceNormalizer
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.split import VariadicSplit
from openvino.tools.mo.front.common.partial_infer.concat import concat_infer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.ops.strided_slice import StridedSlice
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, valued_const_with_data, regular_op_with_empty_data, \
    connect, regular_op, empty_data, regular_op_with_shaped_data

edges = (
    *connect('input', '0:strided_slice'),
    *connect('begin', '1:strided_slice'),
    *connect('end', '2:strided_slice'),
    *connect('strides', '3:strided_slice'),
    *connect('strided_slice', 'res')
)

edges_without_strides = (
    *connect('input', '0:strided_slice'),
    *connect('begin', '1:strided_slice'),
    *connect('end', '2:strided_slice'),
    *connect('strided_slice', 'res')
)


class TestStridedSliceNormalizer(unittest.TestCase):

    def test_strided_slice_extend_inputs(self):
        input_shape = (16, 100, 100, 3)
        nodes = {
            **valued_const_with_data('input', np.arange(np.product(input_shape)).reshape(*input_shape)),
            **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice',
                                                           'type': 'StridedSlice',
                                                           'begin_mask': [1, 1, 1],
                                                           'end_mask': [1, 1, 1],
                                                           'shrink_axis_mask': [0, 0, 0],
                                                           'new_axis_mask': [0, 0, 0],
                                                           'ellipsis_mask': [0, 0, 0],
                                                           'infer': StridedSlice.infer}),

            **regular_op_with_empty_data('strided_slice_ref', {'op': 'StridedSlice',
                                                               'type': 'StridedSlice',
                                                               'begin_mask': [1, 1, 1, 0],
                                                               'end_mask': [1, 1, 1, 0],
                                                               'new_axis_mask': [0, 0, 0, 0],
                                                               'shrink_axis_mask': [0, 0, 0, 0],
                                                               'ellipsis_mask': [0, 0, 0, 0],
                                                               'infer': StridedSlice.infer}),
            **valued_const_with_data('begin', int64_array([0, 0, 0])),
            **valued_const_with_data('begin_placeholder', int64_array([0])),
            **regular_op_with_empty_data('begin_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **valued_const_with_data('end', int64_array([4, 25, 50])),
            **valued_const_with_data('end_placeholder', int64_array([0])),
            **regular_op_with_empty_data('end_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **valued_const_with_data('strides', int64_array([1, 1, 1])),
            **valued_const_with_data('strides_placeholder', int64_array([1])),
            **regular_op_with_empty_data('strides_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        edges_ref_extended_inputs = (
            *connect('input', '0:strided_slice_ref'),

            *connect('begin', '0:begin_concat'),
            *connect('begin_placeholder', '1:begin_concat'),
            *connect('begin_concat', '1:strided_slice_ref'),

            *connect('end', '0:end_concat'),
            *connect('end_placeholder', '1:end_concat'),
            *connect('end_concat', '2:strided_slice_ref'),

            *connect('strides', '0:strides_concat'),
            *connect('strides_placeholder', '1:strides_concat'),
            *connect('strides_concat', '3:strided_slice_ref'),

            *connect('strided_slice_ref', 'res')
        )

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph_ref = build_graph(nodes, edges_ref_extended_inputs, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph_ref.stage = 'middle'

        graph = partial_infer(graph)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        graph_ref = partial_infer(graph_ref)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, 'Graphs after StridedSliceNormalizer do not match to reference: {}'.format(resp))

    def test_strided_slice_extend_inputs_without_strides(self):
        input_shape = (16, 100, 100, 3)
        nodes = {
            **valued_const_with_data('input', np.arange(np.product(input_shape)).reshape(*input_shape)),
            **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice',
                                                           'type': 'StridedSlice',
                                                           'begin_mask': [1, 1, 1],
                                                           'end_mask': [1, 1, 1],
                                                           'shrink_axis_mask': [1, 0, 0],
                                                           'new_axis_mask': [0, 0, 0],
                                                           'ellipsis_mask': [0, 0, 0],
                                                           'infer': StridedSlice.infer}),

            **regular_op_with_empty_data('strided_slice_ref', {'op': 'StridedSlice',
                                                               'type': 'StridedSlice',
                                                               'begin_mask': [1, 1, 1, 0],
                                                               'end_mask': [1, 1, 1, 0],
                                                               'new_axis_mask': [0, 0, 0, 0],
                                                               'shrink_axis_mask': [1, 0, 0, 0],
                                                               'ellipsis_mask': [0, 0, 0, 0],
                                                               'infer': StridedSlice.infer}),
            **valued_const_with_data('begin', int64_array([0, 0, 0])),
            **valued_const_with_data('begin_placeholder', int64_array([0])),
            **regular_op_with_empty_data('begin_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **valued_const_with_data('end', int64_array([4, 25, 50])),
            **valued_const_with_data('end_placeholder', int64_array([0])),
            **regular_op_with_empty_data('end_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),
            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        edges_ref_extended_inputs = (
            *connect('input', '0:strided_slice_ref'),

            *connect('begin', '0:begin_concat'),
            *connect('begin_placeholder', '1:begin_concat'),
            *connect('begin_concat', '1:strided_slice_ref'),

            *connect('end', '0:end_concat'),
            *connect('end_placeholder', '1:end_concat'),
            *connect('end_concat', '2:strided_slice_ref'),

            *connect('strided_slice_ref', 'res')
        )

        graph = build_graph(nodes, edges_without_strides, nodes_with_edges_only=True)
        graph_ref = build_graph(nodes, edges_ref_extended_inputs, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph_ref.stage = 'middle'

        graph = partial_infer(graph)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        graph_ref = partial_infer(graph_ref)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, 'Graphs after StridedSliceNormalizer do not match to reference: {}'.format(resp))

    def test_strided_slice_unrooll_ellipsis(self):
        input_shape = (10, 10, 10, 10)
        # out = inp[1:4, ..., 0:5] -> inp[1:4, :, :, 0:5] => out_shape = (3, 10, 10, 5)
        ellipsis_start = 1

        nodes = {
            **valued_const_with_data('input', np.arange(np.product(input_shape)).reshape(*input_shape)),
            **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice', 'type': 'StridedSlice',
                                                           'begin_mask': [1, 1, 1], 'end_mask': [1, 1, 1],
                                                           'shrink_axis_mask': [0, 0, 0],
                                                           'new_axis_mask': [0, 0, 0],
                                                           'ellipsis_mask': [0, 1, 0],
                                                           'infer': StridedSlice.infer}),

            **regular_op_with_empty_data('strided_slice_ref', {'op': 'StridedSlice', 'begin_mask': [1, 0, 0, 1],
                                                               'end_mask': [1, 0, 0, 1], 'ellipsis_mask': [0, 0, 0, 0],
                                                               'new_axis_mask': [0, 0, 0, 0],
                                                               'shrink_axis_mask': [0, 0, 0, 0],
                                                               'infer': StridedSlice.infer}),

            **valued_const_with_data('begin', int64_array([1, 0, 0])),
            **valued_const_with_data('split_axis_begin', int64_array(0)),
            **valued_const_with_data('splits_lengths_begin', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_begin', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_begin_data_1'),
            **valued_const_with_data('begin_placeholder', int64_array([0])),
            **regular_op_with_empty_data('begin_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),

            **valued_const_with_data('end', int64_array([4, 0, 5])),
            **valued_const_with_data('split_axis_end', int64_array(0)),
            **valued_const_with_data('splits_lengths_end', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_end', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_end_data_1'),
            **valued_const_with_data('end_placeholder', int64_array([0])),
            **regular_op_with_empty_data('end_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),

            **valued_const_with_data('strides', int64_array([1, 1, 1])),
            **valued_const_with_data('split_axis_strides', int64_array(0)),
            **valued_const_with_data('splits_lengths_strides', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_strides', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_strides_data_1'),
            **valued_const_with_data('strides_placeholder', int64_array([1])),
            **regular_op_with_empty_data('strides_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),

            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        edges_ref_ellipsis_unrolled = (
            *connect('input', '0:strided_slice_ref'),

            *connect('begin', '0:split_for_begin'),
            *connect('split_axis_begin', '1:split_for_begin'),
            *connect('splits_lengths_begin', '2:split_for_begin'),
            *connect('split_for_begin:0', '0:begin_concat'),
            *connect('begin_placeholder', '1:begin_concat'),
            ('split_for_begin', 'split_for_begin_data_1', {'out': 1, 'in': 2}),
            ('split_for_begin_data_1', 'begin_concat', {'out': 1, 'in': 2}),
            *connect('begin_concat', '1:strided_slice_ref'),

            *connect('end', '0:split_for_end'),
            *connect('split_axis_end', '1:split_for_end'),
            *connect('splits_lengths_end', '2:split_for_end'),
            *connect('split_for_end:0', '0:end_concat'),
            *connect('end_placeholder', '1:end_concat'),
            ('split_for_end', 'split_for_end_data_1', {'out': 1, 'in': 2}),
            ('split_for_end_data_1', 'end_concat', {'out': 1, 'in': 2}),
            *connect('end_concat', '2:strided_slice_ref'),

            *connect('strides', '0:split_for_strides'),
            *connect('split_axis_strides', '1:split_for_strides'),
            *connect('splits_lengths_strides', '2:split_for_strides'),
            *connect('split_for_strides:0', '0:strides_concat'),
            *connect('strides_placeholder', '1:strides_concat'),
            ('split_for_strides', 'split_for_strides_data_1', {'out': 1, 'in': 2}),
            ('split_for_strides_data_1', 'strides_concat', {'out': 1, 'in': 2}),
            *connect('strides_concat', '3:strided_slice_ref'),

            *connect('strided_slice_ref', 'res')
        )

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph_ref = build_graph(nodes, edges_ref_ellipsis_unrolled, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph_ref.stage = 'middle'
        graph = partial_infer(graph)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        graph_ref = partial_infer(graph_ref)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, 'Graphs after StridedSliceNormalizer do not match to reference: {}'.format(resp))

    def test_strided_slice_unrooll_ellipsis_without_strides(self):
        input_shape = (10, 10, 10, 10)
        # out = inp[1:4, ..., 0:5] -> inp[1:4, :, :, 0:5] => out_shape = (3, 10, 10, 5)
        ellipsis_start = 1

        nodes = {
            **valued_const_with_data('input', np.arange(np.product(input_shape)).reshape(*input_shape)),
            **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice', 'type': 'StridedSlice',
                                                           'begin_mask': [1, 1, 1], 'end_mask': [1, 1, 1],
                                                           'shrink_axis_mask': [0, 0, 0],
                                                           'new_axis_mask': [0, 0, 0],
                                                           'ellipsis_mask': [0, 1, 0],
                                                           'infer': StridedSlice.infer}),

            **regular_op_with_empty_data('strided_slice_ref', {'op': 'StridedSlice', 'begin_mask': [1, 0, 0, 1],
                                                               'end_mask': [1, 0, 0, 1], 'ellipsis_mask': [0, 0, 0, 0],
                                                               'new_axis_mask': [0, 0, 0, 0],
                                                               'shrink_axis_mask': [0, 0, 0, 0],
                                                               'infer': StridedSlice.infer}),

            **valued_const_with_data('begin', int64_array([1, 0, 0])),
            **valued_const_with_data('split_axis_begin', int64_array(0)),
            **valued_const_with_data('splits_lengths_begin', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_begin', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_begin_data_1'),
            **valued_const_with_data('begin_placeholder', int64_array([0])),
            **regular_op_with_empty_data('begin_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),

            **valued_const_with_data('end', int64_array([4, 0, 5])),
            **valued_const_with_data('split_axis_end', int64_array(0)),
            **valued_const_with_data('splits_lengths_end', int64_array([ellipsis_start, -1])),
            **regular_op_with_empty_data('split_for_end', {'op': 'VariadicSplit', 'infer': VariadicSplit.infer}),
            **empty_data('split_for_end_data_1'),
            **valued_const_with_data('end_placeholder', int64_array([0])),
            **regular_op_with_empty_data('end_concat',
                                         {'op': 'Concat', 'infer': concat_infer, 'axis': 0, 'dim_attrs': {}}),

            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        edges_ref_ellipsis_unrolled = (
            *connect('input', '0:strided_slice_ref'),

            *connect('begin', '0:split_for_begin'),
            *connect('split_axis_begin', '1:split_for_begin'),
            *connect('splits_lengths_begin', '2:split_for_begin'),
            *connect('split_for_begin:0', '0:begin_concat'),
            *connect('begin_placeholder', '1:begin_concat'),
            ('split_for_begin', 'split_for_begin_data_1', {'out': 1, 'in': 2}),
            ('split_for_begin_data_1', 'begin_concat', {'out': 1, 'in': 2}),
            *connect('begin_concat', '1:strided_slice_ref'),

            *connect('end', '0:split_for_end'),
            *connect('split_axis_end', '1:split_for_end'),
            *connect('splits_lengths_end', '2:split_for_end'),
            *connect('split_for_end:0', '0:end_concat'),
            *connect('end_placeholder', '1:end_concat'),
            ('split_for_end', 'split_for_end_data_1', {'out': 1, 'in': 2}),
            ('split_for_end_data_1', 'end_concat', {'out': 1, 'in': 2}),
            *connect('end_concat', '2:strided_slice_ref'),

            *connect('strided_slice_ref', 'res')
        )

        graph = build_graph(nodes, edges_without_strides, nodes_with_edges_only=True)
        graph_ref = build_graph(nodes, edges_ref_ellipsis_unrolled, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph_ref.stage = 'middle'
        graph = partial_infer(graph)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        graph_ref = partial_infer(graph_ref)

        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=False)
        self.assertTrue(flag, 'Graphs after StridedSliceNormalizer do not match to reference: {}'.format(resp))


class TestStridedSliceShapeInferAfterNormalizer(unittest.TestCase):
    # check that after inserting Splits and Concats we still get the same shape

    def run_infer_test(self, inp, ref_res, begin, end, strides, begin_mask, end_mask,
                       shrink_axis_mask, new_axis_mask, ellipsis_mask):
        nodes = {
            **valued_const_with_data('input', np.arange(np.product(inp)).reshape(*inp)),
            **valued_const_with_data('begin', int64_array(begin)),
            **valued_const_with_data('end', int64_array(end)),
            **valued_const_with_data('strides', int64_array(strides)),
            **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice', 'type': 'StridedSlice',
                                                           'begin_mask': begin_mask, 'end_mask': end_mask,
                                                           'shrink_axis_mask': shrink_axis_mask,
                                                           'new_axis_mask': new_axis_mask,
                                                           'ellipsis_mask': ellipsis_mask,
                                                           'infer': StridedSlice.infer}),
            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph = partial_infer(graph)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)

        node = Node(graph, 'strided_slice')
        res = node.out_port(0).data.get_shape()
        npt.assert_array_equal(res, ref_res)

    def test_strided_slice_infer_after_normalizer_1(
            self,  # inp[0, :34, 20, :2]
            inp=(1, 35, 35, 3), ref_res=(34, 2),
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1),
            begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1, 0, 1, 0), new_axis_mask=(0,),
            ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_2(
            self,  # inp[0:3, 0:1, 5:0:-1]
            inp=(10, 10, 10, 10), ref_res=(3, 1, 5, 10),
            begin=(0, 0, 5), end=(3, 1, 0), strides=(1, 1, -1), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_3(
            self,  # inp[1:34, 0, :, :2]
            inp=(1, 35, 35, 3), ref_res=(1, 35, 2),
            begin=(0, 0, 0, 0), end=(1, 34, 0, 2), strides=(1, 1, 1, 1), begin_mask=(1, 1, 0, 0), end_mask=(1, 0, 0, 1),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 0)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_4(
            self,  # inp[1:34, :, :, :2] begin mask is (1,) so only one value can be specified
            inp=(1, 35, 35, 3), ref_res=(1, 35, 2),
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(1, 0, 0,), end_mask=(1, 0, 0, 1),
            shrink_axis_mask=(0, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_5(
            self,  # inp[:, :, :, :] since all begin and end masks are zero
            inp=(1, 35, 35, 3), ref_res=(1, 35, 35, 3),
            begin=(1, 10, 10, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0),
            end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_6(
            self,  # inp[0]
            inp=(1, 35, 35, 3), ref_res=(35, 35, 3),
            begin=(0,), end=(1,), strides=(1,), begin_mask=(1,), end_mask=(0,),
            shrink_axis_mask=(1,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_7(
            self,  # inp[0, 20], ends can be of any value
            inp=(1, 35, 35, 3), ref_res=(35, 3),
            begin=(0, 20), end=(1, 9999), strides=(1, 1), begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_8(
            self,  # inp[0, 0:34, 20:22, new_axis], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(34, 2, 1, 3),
            begin=(0, 0, 20, 0), end=(1, 34, 22, 2), strides=(1, 1, 1, 1), begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1,), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_9(
            self,  # inp[:, 0:4, 20, new_axis], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(1, 4, 1, 3),
            begin=(0, 0, 20, 0), end=(0, 4, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 1, 0, 0), end_mask=(0, 1, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_10(
            self,  # inp[:, 0:4, new_axis, 20], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(1, 4, 1, 3),
            begin=(0, 0, 0, 20), end=(0, 4, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 1, 0, 0), end_mask=(0, 1, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_11(
            self,  # inp[0, :, 0:34, 20:22, new_axis], both new_axis and shrink_axis are present
            inp=(1, 3, 35, 35), ref_res=(3, 34, 2, 1),
            begin=(0, 0, 0, 20, 0), end=(1, 0, 34, 22, 0), strides=(1, 1, 1, 1, 1),
            begin_mask=(1, 0, 1, 1, 1), end_mask=(1, 0, 1, 1, 1),
            shrink_axis_mask=(1,), new_axis_mask=(0, 0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_12(
            self,  # inp[0, :34, 20, :2]
            inp=(1, 35, 35, 3), ref_res=(34, 2),
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0, 1, 1, 1),
            end_mask=(0, 1, 1, 1),
            shrink_axis_mask=(1, 0, 1, 0), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_13(
            self,  # inp[0, 0, 0], since it's shrink_axis ends can be of any value
            inp=(1, 35, 35, 3), ref_res=(3,),
            begin=(0, 0, 0), end=(1, 34444, 20), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(1, 1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_14(
            self,  # inp[0, 0, 0], since begin_mask is [0], begin can be of any value
            inp=(1, 35, 35, 3), ref_res=(1, 18, 18, 3),
            begin=(0, 0, 0), end=(1, 35, 35), strides=(2, 2, 2), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0, 0, 0), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    # with ellipsis
    def test_strided_slice_infer_after_normalizer_15(
            self,  # inp[..., np.newaxis]
            inp=(1, 35, 35), ref_res=(1, 35, 35, 1),
            begin=(101, 0), end=(0, 0), strides=(-1, -1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_16(
            self,  # inp_shape = (1, 720, 1080), out = inp[..., :100, None] => out_shape = (1, 720, 100, 1)
            inp=(1, 720, 1080), ref_res=(1, 720, 100, 1),
            begin=(0, 0, 0), end=(0, 100, 0), strides=(1, 1, 1), begin_mask=(0, 1, 0), end_mask=(0, 1, 0),
            shrink_axis_mask=(0,), new_axis_mask=(0, 0, 1), ellipsis_mask=(1,)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_17(
            self,  # inp_shape = (1, 720, 1080, 3), out = inp[..., :-1] => out_shape = (1, 720, 100, 2)
            inp=(1, 720, 1080, 3), ref_res=(1, 720, 1080, 2),
            begin=(0, 0), end=(0, -1), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 1),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_infer_after_normalizer_18(
            self,  # inp_shape = (1, 720, 1080, 3), out = inp[..., 2] => out_shape = (1, 720, 1080)
            inp=(1, 720, 1080, 3), ref_res=(1, 720, 1080),
            begin=(0, 2), end=(0, 0), strides=(1, 1), begin_mask=(0, 1), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
    ):
        self.run_infer_test(inp, ref_res, begin, end, strides,
                            begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    # automatically generated the whole range of 2d slices over 2d, 3d and 4d input tensors
    def test_normalizer_auto_infer_strided_slice_2d_over_2d_0(self):
        """
        inp_shape = (1, 100), out = inp[:, :] => out_shape = (1, 100)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_1(self):
        """
        inp_shape = (1, 100), out = inp[:, None] => out_shape = (1, 1, 100)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_2(self):
        """
        inp_shape = (1, 100), out = inp[:, 0] => out_shape = (1,)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_3(self):
        """
        inp_shape = (1, 100), out = inp[..., :] => out_shape = (1, 100)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_4(self):
        """
        inp_shape = (1, 100), out = inp[..., None] => out_shape = (1, 100, 1)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 100, 1),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_5(self):
        """
        inp_shape = (1, 100), out = inp[..., 0] => out_shape = (1,)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_6(self):
        """
        inp_shape = (1, 100), out = inp[None, :] => out_shape = (1, 1, 100)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_7(self):
        """
        inp_shape = (1, 100), out = inp[None, None] => out_shape = (1, 1, 1, 100)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 1, 1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_8(self):
        """
        inp_shape = (1, 100), out = inp[None, 0] => out_shape = (1, 100)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_9(self):
        """
        inp_shape = (1, 100), out = inp[0, :] => out_shape = (100,)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(100,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_10(self):
        """
        inp_shape = (1, 100), out = inp[0, None] => out_shape = (1, 100)
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_2d_11(self):
        """
        inp_shape = (1, 100), out = inp[0, 0] => out_shape = ()
        """
        self.run_infer_test(
            inp=(1, 100), ref_res=(),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_0(self):
        """
        inp_shape = (1, 100, 200), out = inp[:, :] => out_shape = (1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_1(self):
        """
        inp_shape = (1, 100, 200), out = inp[:, None] => out_shape = (1, 1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_2(self):
        """
        inp_shape = (1, 100, 200), out = inp[:, 0] => out_shape = (1, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_3(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., :] => out_shape = (1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_4(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., None] => out_shape = (1, 100, 200, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 200, 1),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_5(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., 0] => out_shape = (1, 100)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_6(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, :] => out_shape = (1, 1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_7(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, None] => out_shape = (1, 1, 1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 1, 1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_8(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, 0] => out_shape = (1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_9(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, :] => out_shape = (100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_10(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, None] => out_shape = (1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_3d_11(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, 0] => out_shape = (200,)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(200,),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_0(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, :] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_1(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, None] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_2(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, 0] => out_shape = (1, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_3(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_4(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None] => out_shape = (1, 100, 200, 3, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3, 1),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_5(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0] => out_shape = (1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_6(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_7(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, None] => out_shape = (1, 1, 1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_8(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_9(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, :] => out_shape = (100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_10(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_2d_over_4d_11(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, 0] => out_shape = (200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    # automatically generated slices from 3d to 5d d input tensors
    # fixed number of ellipsis, newaxis and shrink_axis
    def test_normalizer_auto_infer_strided_slice_3d_over_3d_0(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, ..., 0] => out_shape = (1, 1, 100)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 1, 100),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 0, 1), new_axis_mask=(1, 0, 0), ellipsis_mask=(0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_3d_1(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., None, 0] => out_shape = (1, 100, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 1),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 0, 1), new_axis_mask=(0, 1, 0), ellipsis_mask=(1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_3d_2(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, None, ...] => out_shape = (1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 200),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(1, 0, 0), new_axis_mask=(0, 1, 0), ellipsis_mask=(0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_3d_3(self):
        """
        inp_shape = (1, 100, 200), out = inp[0, ..., None] => out_shape = (100, 200, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(100, 200, 1),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(1, 0, 0), new_axis_mask=(0, 0, 1), ellipsis_mask=(0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_3d_4(self):
        """
        inp_shape = (1, 100, 200), out = inp[None, 0, ...] => out_shape = (1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 200),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 1, 0), new_axis_mask=(1, 0, 0), ellipsis_mask=(0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_3d_5(self):
        """
        inp_shape = (1, 100, 200), out = inp[..., 0, None] => out_shape = (1, 100, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200), ref_res=(1, 100, 1),
            begin=(0, 0, 0), end=(0, 0, 0), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(0, 1, 0), new_axis_mask=(0, 0, 1), ellipsis_mask=(1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_4d_0(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, ..., 0, :] => out_shape = (1, 1, 100, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_4d_1(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None, 0, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_4d_2(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_4d_3(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, ..., None, :] => out_shape = (100, 200, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(100, 200, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_4d_4(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_4d_5(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0, None, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_5d_0(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, ..., 0, :, :] => out_shape = (1, 1, 100, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 100, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_5d_1(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., None, 0, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_5d_2(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, None, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_5d_3(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, ..., None, :, :] => out_shape = (100, 200, 1, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(100, 200, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_5d_4(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, 0, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_3d_over_5d_5(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., 0, None, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_0(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, ..., 0, :] => out_shape = (1, 1, 100, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_1(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None, 0, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_2(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_3(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, ..., None, :] => out_shape = (100, 200, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(100, 200, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_4(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0, ..., :] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_5(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0, None, :] => out_shape = (1, 100, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_6(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, ..., :, 0] => out_shape = (1, 1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_7(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., None, :, 0] => out_shape = (1, 100, 1, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 1, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_8(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, None, :, ...] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_9(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, ..., :, None] => out_shape = (100, 200, 3, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(100, 200, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_10(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, 0, :, ...] => out_shape = (1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_11(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., 0, :, None] => out_shape = (1, 100, 3, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_12(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, :, ..., 0] => out_shape = (1, 1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_13(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., :, None, 0] => out_shape = (1, 100, 200, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_14(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, :, None, ...] => out_shape = (100, 1, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(100, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_15(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[0, :, ..., None] => out_shape = (100, 200, 3, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(100, 200, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_16(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[None, :, 0, ...] => out_shape = (1, 1, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_17(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[..., :, 0, None] => out_shape = (1, 100, 200, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_18(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, None, ..., 0] => out_shape = (1, 1, 100, 200)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_19(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, ..., None, 0] => out_shape = (1, 100, 200, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_20(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, 0, None, ...] => out_shape = (1, 1, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_21(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, 0, ..., None] => out_shape = (1, 200, 3, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 200, 3, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_22(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, None, 0, ...] => out_shape = (1, 1, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 200, 3),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_4d_23(self):
        """
        inp_shape = (1, 100, 200, 3), out = inp[:, ..., 0, None] => out_shape = (1, 100, 200, 1)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 1),
            begin=(0, 0, 0, 0), end=(0, 0, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0), end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_0(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, ..., 0, :, :] => out_shape = (1, 1, 100, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 100, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_1(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., None, 0, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_2(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, None, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_3(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, ..., None, :, :] => out_shape = (100, 200, 1, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(100, 200, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_4(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, 0, ..., :, :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_5(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., 0, None, :, :] => out_shape = (1, 100, 1, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 1, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_6(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, ..., :, 0, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_7(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., None, :, 0, :] => out_shape = (1, 100, 1, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 1, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_8(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, None, :, ..., :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_9(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, ..., :, None, :] => out_shape = (100, 200, 10, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(100, 200, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_10(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, 0, :, ..., :] => out_shape = (1, 100, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_11(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., 0, :, None, :] => out_shape = (1, 100, 10, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_12(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, :, ..., 0, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_13(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., :, None, 0, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_14(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, :, None, ..., :] => out_shape = (100, 1, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(100, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_15(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[0, :, ..., None, :] => out_shape = (100, 200, 10, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(100, 200, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(1, 0, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_16(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[None, :, 0, ..., :] => out_shape = (1, 1, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(1, 0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_17(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[..., :, 0, None, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(1, 0, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_18(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, None, ..., 0, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_19(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, ..., None, 0, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_20(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, 0, None, ..., :] => out_shape = (1, 1, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 1, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_21(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, 0, ..., None, :] => out_shape = (1, 200, 10, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 200, 10, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 1, 0, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 0, 1, 0, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_22(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, None, 0, ..., :] => out_shape = (1, 1, 200, 10, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 1, 200, 10, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 1, 0, 0, 0), ellipsis_mask=(0, 0, 0, 1, 0)
        )

    def test_normalizer_auto_infer_strided_slice_4d_over_5d_23(self):
        """
        inp_shape = (1, 100, 200, 10, 3), out = inp[:, ..., 0, None, :] => out_shape = (1, 100, 200, 1, 3)
        """
        self.run_infer_test(
            inp=(1, 100, 200, 10, 3), ref_res=(1, 100, 200, 1, 3),
            begin=(0, 0, 0, 0, 0), end=(0, 0, 0, 0, 0), strides=(1, 1, 1, 1, 1), begin_mask=(0, 0, 0, 0, 0),
            end_mask=(0, 0, 0, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0, 0), new_axis_mask=(0, 0, 0, 1, 0), ellipsis_mask=(0, 1, 0, 0, 0)
        )


class TestStridedSlicePermute(unittest.TestCase):
    def run_permute_test(self, inp, ref_res, begin, end, strides, begin_mask, end_mask,
                         shrink_axis_mask, new_axis_mask, ellipsis_mask):
        from openvino.tools.mo.middle.ApplyPermutations import ApplyPermutation
        from openvino.tools.mo.middle.MergeNodesPermutations import MergeNodesPermutations
        from openvino.tools.mo.middle.ApplyNHWCtoNCHWpermutation import ApplyNHWCtoNCHWpermutation
        from openvino.tools.mo.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose
        from openvino.tools.mo.middle.MarkSubgraphsWithCorrectLayout import MarkSubGraphsWithCorrectLayout
        nodes = {
            **regular_op_with_shaped_data('input', int64_array(inp), {'op': 'Parameter', 'type': 'Parameter',
                                                                      # need to specify shape in 2 places
                                                                      'shape': int64_array(inp),
                                                                      'infer': Parameter.infer}),
            **valued_const_with_data('begin', int64_array(begin)),
            **valued_const_with_data('end', int64_array(end)),
            **valued_const_with_data('strides', int64_array(strides)),
            **regular_op_with_empty_data('strided_slice',
                                         {'op': 'StridedSlice', 'type': 'StridedSlice',  # need for permute
                                          'begin_mask': begin_mask, 'end_mask': end_mask,
                                          'shrink_axis_mask': shrink_axis_mask,
                                          'new_axis_mask': new_axis_mask,
                                          'ellipsis_mask': ellipsis_mask,
                                          'infer': StridedSlice.infer}),
            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph.graph['layout'] = 'NHWC'

        graph = partial_infer(graph)
        StridedSliceNormalizer().find_and_replace_pattern(graph)
        graph = partial_infer(graph)
        MarkSubGraphsWithCorrectLayout().find_and_replace_pattern(graph)
        InsertLayoutPropagationTranspose().find_and_replace_pattern(graph)
        ApplyNHWCtoNCHWpermutation().find_and_replace_pattern(graph)
        MergeNodesPermutations().find_and_replace_pattern(graph)
        ApplyPermutation().find_and_replace_pattern(graph)
        graph = partial_infer(graph)

        node = Node(graph, 'strided_slice')
        res = node.out_port(0).data.get_shape()
        npt.assert_array_equal(res, ref_res)

    def test_strided_slice_permute_1(
            self,  # inp[0, :34, 20, :2]
            inp=(1, 35, 35, 3), ref_res=(34, 2),
            begin=(0, 0, 20, 0), end=(1, 34, 21, 2), strides=(1, 1, 1, 1),
            begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1, 0, 1, 0), new_axis_mask=(0,),
            ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_2(
            self,  # inp[0:3, 0:1, 5:0:-1]
            inp=(10, 10, 10, 10), ref_res=(3, 10, 1, 5),
            begin=(0, 0, 5), end=(3, 1, 0), strides=(1, 1, -1), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_3(
            self,  # inp[1:34, 0, :, :2]
            inp=(1, 35, 35, 3), ref_res=(1, 35, 2),
            begin=(0, 0, 0, 0), end=(1, 34, 0, 2), strides=(1, 1, 1, 1), begin_mask=(1, 1, 0, 0), end_mask=(1, 0, 0, 1),
            shrink_axis_mask=(0, 1, 0, 0), new_axis_mask=(0, 0, 0, 0), ellipsis_mask=(0, 0, 0, 0)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_4(
            # no shrink/new axis therefore will be permuted
            # inp[0:1, :, :, :2] begin mask is (1,) so only one begin value need to be specified
            self,
            inp=(16, 35, 35, 3), ref_res=(1, 2, 35, 35),
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(1, 0, 0,), end_mask=(1, 0, 0, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_5(
            self,  # inp[:, :, :, :] since all begin and end masks are zero
            inp=(1, 35, 35, 3), ref_res=(1, 3, 35, 35),
            begin=(1, 10, 10, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0, 0, 0, 0),
            end_mask=(0, 0, 0, 0),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_6(
            self,  # inp[0]
            inp=(1, 35, 35, 3), ref_res=(35, 35, 3),
            begin=(0,), end=(1,), strides=(1,), begin_mask=(1,), end_mask=(0,),
            shrink_axis_mask=(1,), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_7(
            self,  # inp[0, 20], ends can be of any value
            inp=(1, 35, 35, 3), ref_res=(35, 3),
            begin=(0, 20), end=(1, 9999), strides=(1, 1), begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_8(
            self,  # inp[0, 0:34, 20:22, new_axis], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(34, 2, 1, 3),
            begin=(0, 0, 20, 0), end=(1, 34, 22, 2), strides=(1, 1, 1, 1), begin_mask=(0,), end_mask=(0,),
            shrink_axis_mask=(1,), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_9(
            self,  # inp[:, 0:4, 20, new_axis], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(1, 4, 1, 3),
            begin=(0, 0, 20, 0), end=(0, 4, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 1, 0, 0), end_mask=(0, 1, 0, 0),
            shrink_axis_mask=(0, 0, 1, 0), new_axis_mask=(0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_10(
            self,  # inp[:, 0:4, new_axis, 20], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(1, 4, 1, 3),
            begin=(0, 0, 0, 20), end=(0, 4, 0, 0), strides=(1, 1, 1, 1), begin_mask=(0, 1, 0, 0), end_mask=(0, 1, 0, 0),
            shrink_axis_mask=(0, 0, 0, 1), new_axis_mask=(0, 0, 1, 0), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_11(
            self,  # inp[0, :, 0:34, 1:3, new_axis], both new_axis and shrink_axis are present
            inp=(1, 35, 35, 3), ref_res=(35, 34, 2, 1),
            begin=(0, 0, 0, 1, 0), end=(1, 0, 34, 3, 0), strides=(1, 1, 1, 1, 1),
            begin_mask=(1, 0, 1, 1, 1), end_mask=(1, 0, 1, 1, 1),
            shrink_axis_mask=(1, 0, 0, 0), new_axis_mask=(0, 0, 0, 0, 1), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_12(
            self,  # inp[0, :34, 20, :2]
            inp=(1, 35, 35, 3), ref_res=(34, 2),
            begin=(0, 0, 0, 0), end=(1, 34, 20, 2), strides=(1, 1, 1, 1), begin_mask=(0, 1, 1, 1),
            end_mask=(0, 1, 1, 1),
            shrink_axis_mask=(1, 0, 1, 0), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_13(
            self,  # inp[0, 0, 0], since it's shrink_axis ends can be of any value
            inp=(1, 35, 35, 3), ref_res=(3,),
            begin=(0, 0, 0), end=(1, 34444, 20), strides=(1, 1, 1), begin_mask=(0, 0, 0), end_mask=(0, 0, 0),
            shrink_axis_mask=(1, 1, 1), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_14(
            self,  # inp[0, 0, 0], since begin_mask is [0], begin can be of any value
            inp=(1, 35, 35, 3), ref_res=(1, 3, 18, 18),
            begin=(0, 0, 0), end=(1, 35, 35), strides=(2, 2, 2), begin_mask=(1, 1, 1), end_mask=(1, 1, 1),
            shrink_axis_mask=(0, 0, 0), new_axis_mask=(0,), ellipsis_mask=(0,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

        # with ellipsis

    def test_strided_slice_permute_15(
            self,  # inp[..., np.newaxis]
            inp=(1, 35, 35), ref_res=(1, 35, 35, 1),
            begin=(101, 0), end=(0, 0), strides=(-1, -1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_16(
            self,  # inp_shape = (1, 720, 1080), out = inp[..., :100, None] => out_shape = (1, 720, 100, 1)
            inp=(1, 720, 1080), ref_res=(1, 720, 100, 1),
            begin=(0, 0, 0), end=(0, 100, 0), strides=(1, 1, 1), begin_mask=(0, 1, 0), end_mask=(0, 1, 0),
            shrink_axis_mask=(0,), new_axis_mask=(0, 0, 1), ellipsis_mask=(1,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_17(
            self,  # inp_shape = (1, 720, 1080, 3), out = inp[..., :-1] => out_shape = (1, 720, 100, 2)
            inp=(1, 720, 1080, 3), ref_res=(1, 2, 720, 1080),
            begin=(0, 0), end=(0, -1), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 1),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_18(
            self,  # inp_shape = (1, 720, 1080, 3), out = inp[..., 2] => out_shape = (1, 720, 1080)
            inp=(1, 720, 1080, 3), ref_res=(1, 720, 1080),
            begin=(0, 2), end=(0, 0), strides=(1, 1), begin_mask=(0, 1), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    def test_strided_slice_permute_19(
            self,  # inp_shape = (1, 720, 1080, 3), out = input[..., 0:10, 0:3] => out_shape = (1, 720, 10, 3)
            inp=(1, 720, 1080, 3), ref_res=(1, 3, 720, 10),
            begin=(0, 0, 0), end=(0, 10, 3), strides=(1, 1, 1), begin_mask=(0, 1, 1), end_mask=(0, 1, 1),
            shrink_axis_mask=(0,), new_axis_mask=(0,), ellipsis_mask=(1,)
    ):
        self.run_permute_test(inp, ref_res, begin, end, strides,
                              begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask)

    # automatically generated permutation  tests
    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_0(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[:, :],
        out_nchw = inp[:, :, :, :] => out_shape = (1, 3, 100, 200)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 3, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_1(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[:, None] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_2(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[:, 0] => out_shape = (1, 200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_3(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[..., :],
        out_nchw = inp[:, :, :, ...] => out_shape = (1, 3, 100, 200)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 3, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_4(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[..., None] => out_shape = (1, 100, 200, 3, 1)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3, 1),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(0, 1), ellipsis_mask=(1, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_5(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[..., 0],
        out_nchw = inp[:, 0, :, ...] => out_shape = (1, 100, 200)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(0, 0), ellipsis_mask=(1, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_6(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[None, :] => out_shape = (1, 1, 100, 200, 3)
        out_nchw = inp[None, :, :, :, :] => out_shape = (1, 1, 100, 200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_7(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[None, None] => out_shape = (1, 1, 1, 100, 200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 1, 1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 0), new_axis_mask=(1, 1), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_8(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC, (1, 3, 100, 200) in NCHW,
        out_nhwc = inp[None, 0] => out_shape = (1, 100, 200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(0, 1), new_axis_mask=(1, 0), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_9(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC
        out_nhwc = inp[0, :] => out_shape = (100, 200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_10(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC
        out_nhwc = inp[0, None] => out_shape = (1, 100, 200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(1, 100, 200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 1), ellipsis_mask=(0, 0)
        )

    def test_permute_auto_infer_strided_slice_2d_slice_over_4d_11(self):
        """
        inp_shape = (1, 100, 200, 3) in NHWC
        out_nhwc = inp[0, 0] => out_shape = (200, 3)
        """
        self.run_permute_test(
            inp=(1, 100, 200, 3), ref_res=(200, 3),
            begin=(0, 0), end=(0, 0), strides=(1, 1), begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0)
        )


class TestStridedSliceMaskAlignment(unittest.TestCase):
    def run_align_test(self, inp, begin, end, strides,
                       begin_mask, end_mask, shrink_axis_mask, new_axis_mask, ellipsis_mask,
                       begin_mask_ref, end_mask_ref, shrink_axis_mask_ref, new_axis_mask_ref, ellipsis_mask_ref):
        nodes = {
            **regular_op_with_shaped_data('input', int64_array(inp), {'op': 'Parameter', 'type': 'Parameter',
                                                                      # need to specify shape in 2 places
                                                                      'shape': int64_array(inp),
                                                                      'infer': Parameter.infer}),
            **valued_const_with_data('begin', int64_array(begin)),
            **valued_const_with_data('end', int64_array(end)),
            **valued_const_with_data('strides', int64_array(strides)),
            **regular_op_with_empty_data('strided_slice',
                                         {'op': 'StridedSlice', 'type': 'StridedSlice',  # need for permute
                                          'begin_mask': begin_mask, 'end_mask': end_mask,
                                          'shrink_axis_mask': shrink_axis_mask,
                                          'new_axis_mask': new_axis_mask,
                                          'ellipsis_mask': ellipsis_mask}),
            **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
        }

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph.graph['layout'] = 'NHWC'

        nodes_ref = nodes.copy()
        nodes_ref.update({
            **regular_op_with_empty_data('strided_slice',
                                         {'op': 'StridedSlice', 'type': 'StridedSlice',  # need for permute
                                          'begin_mask': begin_mask_ref, 'end_mask': end_mask_ref,
                                          'shrink_axis_mask': shrink_axis_mask_ref,
                                          'new_axis_mask': new_axis_mask_ref,
                                          'ellipsis_mask': ellipsis_mask_ref}),
        })

        graph_ref = build_graph(nodes_ref, edges, nodes_with_edges_only=True)
        res, msg = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)
        assert res, msg

    def test_mask_align_compare_graphs_1(self):
        self.run_align_test(
            inp=(1, 100, 200, 3), begin=(0, 0), end=(0, 0), strides=(1, 1),
            begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0),
            begin_mask_ref=(0, 0), end_mask_ref=(0, 0),
            shrink_axis_mask_ref=(1, 1), new_axis_mask_ref=(0, 0), ellipsis_mask_ref=(0, 0),

        )

    def test_mask_align_compare_graphs_2(self):
        # begin_masks have different values, but they alight to the same mask
        self.run_align_test(
            inp=(1, 100, 200, 3), begin=(0, 0), end=(0, 0), strides=(1, 1),
            begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0),
            begin_mask_ref=(0), end_mask_ref=(0, 0),
            shrink_axis_mask_ref=(1, 1), new_axis_mask_ref=(0, 0), ellipsis_mask_ref=(0, 0),

        )

    def test_mask_align_compare_graphs_3(self):
        # begin_masks have different values, but they alight to the same mask
        self.run_align_test(
            inp=(1, 100, 200, 3), begin=(0, 0), end=(0, 0), strides=(1, 1),
            begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0),
            begin_mask_ref=(0, 0), end_mask_ref=(0),
            shrink_axis_mask_ref=(1, 1), new_axis_mask_ref=(0), ellipsis_mask_ref=(0),
        )

    def test_mask_align_compare_graphs_4(self):
        # begin_masks have different values, but they alight to the same mask
        self.run_align_test(
            inp=(1, 100, 200, 3), begin=(0, 0), end=(0, 0), strides=(1, 1),
            begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 0), new_axis_mask=(0, 0), ellipsis_mask=(0, 0),
            begin_mask_ref=(0, 0), end_mask_ref=(0),
            shrink_axis_mask_ref=(1), new_axis_mask_ref=(0), ellipsis_mask_ref=(0),
        )

    # corner case with and empty slice
    def test_mask_align_compare_graphs_5(self):
        self.run_align_test(
            inp=(1, 100, 200, 3), begin=[], end=[], strides=[],
            begin_mask=[], end_mask=[],
            shrink_axis_mask=[], new_axis_mask=[], ellipsis_mask=[],
            begin_mask_ref=[], end_mask_ref=[],
            shrink_axis_mask_ref=[], new_axis_mask_ref=[], ellipsis_mask_ref=[]
        )

    # emppty mask [] should be aligned into the length of begin
    def test_mask_align_compare_graphs_6(self):
        # begin_masks have different values, but they alight to the same mask
        self.run_align_test(
            inp=(1, 100, 200, 3), begin=(0, 0), end=(0, 0), strides=(1, 1),
            begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0),
            begin_mask_ref=[], end_mask_ref=(0, 0),
            shrink_axis_mask_ref=(1, 1), new_axis_mask_ref=(0, 0), ellipsis_mask_ref=(0, 0),

        )

    # empty mask "" should be transformed into [] and then aligned into the length of begin
    def test_mask_align_compare_graphs_7(self):
        # begin_masks have different values, but they alight to the same mask
        self.run_align_test(
            inp=(1, 100, 200, 3), begin=(0, 0), end=(0, 0), strides=(1, 1),
            begin_mask=(0, 0), end_mask=(0, 0),
            shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0),
            begin_mask_ref="", end_mask_ref=(0, 0),
            shrink_axis_mask_ref=(1, 1), new_axis_mask_ref=(0, 0), ellipsis_mask_ref=(0, 0),

        )

    # negative test
    def test_negative_mask_align_compare_graphs(self):
        with self.assertRaisesRegex(AssertionError, 'have different attr "begin_mask"'):
            self.run_align_test(
                inp=(1, 100, 200, 3), begin=(0, 0), end=(0, 0), strides=(1, 1),
                begin_mask=(0, 0), end_mask=(0, 0),
                shrink_axis_mask=(1, 1), new_axis_mask=(0, 0), ellipsis_mask=(0, 0),
                begin_mask_ref=(0, 1), end_mask_ref=(0, 0),
                shrink_axis_mask_ref=(1, 1), new_axis_mask_ref=(0, 0), ellipsis_mask_ref=(0, 0),
            )
