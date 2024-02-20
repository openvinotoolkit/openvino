# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.SSliceComplex import SSliceComplex
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, connect, \
    regular_op_with_shaped_data, valued_const_with_data

graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', int64_array([3, 100, 100, 2]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('strided_slice_real', int64_array([3, 100, 100]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0, 1]),
            'slices': np.array([Ellipsis, 0])
        }),
    **regular_op_with_shaped_data('strided_slice_imag', int64_array([3, 100, 100]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0, 1]),
            'slices': np.array([Ellipsis, 1])
        }),
    **regular_op_with_shaped_data('complex', int64_array([3, 100, 100, 2]), {'op': 'Complex'}),
    **valued_const_with_data('real_begin', int64_array([0, 0])),
    **valued_const_with_data('imag_begin', int64_array([0, 1])),
    **valued_const_with_data('real_end', int64_array([0, 1])),
    **valued_const_with_data('imag_end', int64_array([0, 2])),
    **valued_const_with_data('real_strides', int64_array([1, 1])),
    **valued_const_with_data('imag_strides', int64_array([1, 1])),
    **regular_op_with_shaped_data('abs', int64_array([3, 100, 100, 2]), {'type': 'Abs', 'op': 'Abs'}),
    **result('output'),
}

graph_edges = [
    ('placeholder', 'placeholder_d', {'out': 0}),
    ('placeholder_d', 'strided_slice_real', {'out': 0, 'in': 0}),
    ('placeholder_d', 'strided_slice_imag', {'out': 0, 'in': 0}),
    *connect('strided_slice_real:0', '0:complex'),
    *connect('strided_slice_imag:0', '1:complex'),
    *connect('real_begin:0', '1:strided_slice_real'),
    *connect('imag_begin:0', '1:strided_slice_imag'),
    *connect('real_end:0', '2:strided_slice_real'),
    *connect('imag_end:0', '2:strided_slice_imag'),
    *connect('real_strides:0', '3:strided_slice_real'),
    *connect('imag_strides:0', '3:strided_slice_imag'),
    *connect('complex:0', '0:abs'),
    *connect('abs:0', 'output'),
]


ref_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder', int64_array([3, 100, 100, 2]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('abs', int64_array([3, 100, 100, 2]), {'type': 'Abs', 'op': 'Abs'}),
    **result('output'),
}

ref_graph_edges = [
    *connect('placeholder:0', '0:abs'),
    *connect('abs:0', 'output'),
]


non_transformed_graph_node_attrs = {
    **regular_op_with_shaped_data('placeholder_0', int64_array([3, 100, 100, 2]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('placeholder_1', int64_array([3, 100, 100, 2]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('strided_slice_real', int64_array([3, 100, 100]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0, 1]),
            'slices': np.array([Ellipsis, 0])
        }),
    **regular_op_with_shaped_data('strided_slice_imag', int64_array([3, 100, 100]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
            'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
            'shrink_axis_mask': int64_array([0, 1]),
            'slices': np.array([Ellipsis, 1])
        }),
    **regular_op_with_shaped_data('complex', int64_array([3, 100, 100, 2]), {'op': 'Complex'}),
    **valued_const_with_data('real_begin', int64_array([0, 0])),
    **valued_const_with_data('imag_begin', int64_array([0, 1])),
    **valued_const_with_data('real_end', int64_array([0, 1])),
    **valued_const_with_data('imag_end', int64_array([0, 2])),
    **valued_const_with_data('real_strides', int64_array([1, 1])),
    **valued_const_with_data('imag_strides', int64_array([1, 1])),
    **regular_op_with_shaped_data('abs', int64_array([3, 100, 100, 2]), {'type': 'Abs', 'op': 'Abs'}),
    **result('output'),
}

non_transformed_graph_edges = [
    *connect('placeholder_0:0', '0:strided_slice_real'),
    *connect('placeholder_1:0', '0:strided_slice_imag'),
    *connect('strided_slice_real:0', '0:complex'),
    *connect('strided_slice_imag:0', '1:complex'),
    *connect('real_begin:0', '1:strided_slice_real'),
    *connect('imag_begin:0', '1:strided_slice_imag'),
    *connect('real_end:0', '2:strided_slice_real'),
    *connect('imag_end:0', '2:strided_slice_imag'),
    *connect('real_strides:0', '3:strided_slice_real'),
    *connect('imag_strides:0', '3:strided_slice_imag'),
    *connect('complex:0', '0:abs'),
    *connect('abs:0', 'output'),
]


graph_node_attrs_2 = {
    **regular_op_with_shaped_data('placeholder', int64_array([3, 100, 2, 66, 34]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('strided_slice_real', int64_array([3, 100, 66, 34]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice',
            'begin_mask': int64_array([0, 0, 1, 0, 0]),
            'end_mask': int64_array([0, 0, 1, 0, 0]),
            'ellipsis_mask': int64_array([0, 0, 0, 0, 0]),
            'new_axis_mask': int64_array([0, 0, 0, 0, 0]),
            'shrink_axis_mask': int64_array([0, 0, 1, 0, 0]),
            'slices': np.array([slice(None, None, 1),
                                slice(None, None, 1),
                                0,
                                slice(None, None, 1),
                                slice(None, None, 1)])
        }),
    **regular_op_with_shaped_data('strided_slice_imag', int64_array([3, 100, 66, 34]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice',
            'begin_mask': int64_array([0, 0, 1, 0, 0]),
            'end_mask': int64_array([0, 0, 1, 0, 0]),
            'ellipsis_mask': int64_array([0, 0, 0, 0, 0]),
            'new_axis_mask': int64_array([0, 0, 0, 0, 0]),
            'shrink_axis_mask': int64_array([0, 0, 1, 0, 0]),
            'slices': np.array([slice(None, None, 1),
                                slice(None, None, 1),
                                1,
                                slice(None, None, 1),
                                slice(None, None, 1)])
        }),
    **regular_op_with_shaped_data('complex', int64_array([3, 100, 66, 34, 2]), {'op': 'Complex'}),
    **valued_const_with_data('real_begin', int64_array([0, 0, 0, 0, 0])),
    **valued_const_with_data('imag_begin', int64_array([0, 0, 1, 0, 0])),
    **valued_const_with_data('real_end', int64_array([0, 0, 1, 0, 0])),
    **valued_const_with_data('imag_end', int64_array([0, 0, 2, 0, 0])),
    **valued_const_with_data('real_strides', int64_array([1, 1, 1, 1, 1])),
    **valued_const_with_data('imag_strides', int64_array([1, 1, 1, 1, 1])),
    **regular_op_with_shaped_data('abs', int64_array([3, 100, 66, 34, 2]), {'type': 'Abs', 'op': 'Abs'}),
    **result('output'),
}


ref_graph_node_attrs_2 = {
    **regular_op_with_shaped_data('placeholder', int64_array([3, 100, 2, 66, 34]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **valued_const_with_data('perm', int64_array([0, 1, 3, 4, 2])),
    **regular_op_with_shaped_data('transpose', int64_array([3, 100, 66, 34, 2]),
                                 {'type': 'Transpose', 'op': 'Transpose'}),
    **regular_op_with_shaped_data('abs', int64_array([3, 100, 66, 34, 2]), {'type': 'Abs', 'op': 'Abs'}),
    **result('output'),
}

ref_graph_edges_2 = [
    *connect('placeholder:0', '0:transpose'),
    *connect('perm:0', '1:transpose'),
    *connect('transpose:0', '0:abs'),
    *connect('abs:0', 'output'),
]


graph_node_attrs_3 = {
    **regular_op_with_shaped_data('placeholder', int64_array([3, 100, 2, 66, 34]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('strided_slice_real', int64_array([3, 100, 66, 34]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice',
            'begin_mask': int64_array([0, 0, 1, 0, 0]),
            'end_mask': int64_array([0, 0, 1, 0, 0]),
            'ellipsis_mask': int64_array([0, 0, 0, 0, 0]),
            'new_axis_mask': int64_array([0, 0, 0, 0, 0]),
            'shrink_axis_mask': int64_array([0, 0, 1, 0, 0]),
            'slices': np.array([slice(None, None, 1),
                                slice(None, None, 1),
                                0,
                                slice(None, None, 1),
                                slice(None, None, 1)])
        }),
    **regular_op_with_shaped_data('strided_slice_imag', int64_array([3, 100, 66, 34]),
        {
            'type': 'StridedSlice', 'op': 'StridedSlice',
            'begin_mask': int64_array([0, 0, 1, 0, 0]),
            'end_mask': int64_array([0, 0, 1, 0, 0]),
            'ellipsis_mask': int64_array([0, 0, 0, 0, 0]),
            'new_axis_mask': int64_array([0, 0, 0, 0, 0]),
            'shrink_axis_mask': int64_array([0, 0, 1, 0, 0]),
            'slices': np.array([slice(None, None, 1),
                                slice(None, None, 1),
                                1,
                                slice(None, None, 1),
                                slice(None, None, 1)])
        }),
    **regular_op_with_shaped_data('complex', int64_array([3, 100, 66, 34, 2]), {'op': 'Complex'}),
    **regular_op_with_shaped_data('roll', int64_array([3, 100, 66, 34, 2]), {'type': 'Roll', 'op': 'Roll'}),
    **valued_const_with_data('real_begin', int64_array([0, 0, 0, 0, 0])),
    **valued_const_with_data('imag_begin', int64_array([0, 0, 1, 0, 0])),
    **valued_const_with_data('real_end', int64_array([0, 0, 1, 0, 0])),
    **valued_const_with_data('imag_end', int64_array([0, 0, 2, 0, 0])),
    **valued_const_with_data('real_strides', int64_array([1, 1, 1, 1, 1])),
    **valued_const_with_data('imag_strides', int64_array([1, 1, 1, 1, 1])),
    **regular_op_with_shaped_data('abs', int64_array([3, 100, 66, 34, 2]), {'type': 'Abs', 'op': 'Abs'}),
    **valued_const_with_data('shift', int64_array([20, 20])),
    **valued_const_with_data('axis', int64_array([1, -2, -1])),
    **result('output'),
}

graph_edges_2 = [
    ('placeholder', 'placeholder_d', {'out': 0}),
    ('placeholder_d', 'strided_slice_real', {'out': 0, 'in': 0}),
    ('placeholder_d', 'strided_slice_imag', {'out': 0, 'in': 0}),
    *connect('strided_slice_real:0', '0:complex'),
    *connect('strided_slice_imag:0', '1:complex'),
    *connect('real_begin:0', '1:strided_slice_real'),
    *connect('imag_begin:0', '1:strided_slice_imag'),
    *connect('real_end:0', '2:strided_slice_real'),
    *connect('imag_end:0', '2:strided_slice_imag'),
    *connect('real_strides:0', '3:strided_slice_real'),
    *connect('imag_strides:0', '3:strided_slice_imag'),
    *connect('complex:0', '0:roll'),
    *connect('shift:0', '1:roll'),
    *connect('axis:0', '2:roll'),
    *connect('roll:0', '0:abs'),
    *connect('abs:0', 'output'),
]

ref_graph_node_attrs_3 = {
    **regular_op_with_shaped_data('placeholder', int64_array([3, 100, 2, 66, 34]),
                                  {'type': 'Parameter', 'op': 'Parameter'}),
    **valued_const_with_data('perm', int64_array([0, 1, 3, 4, 2])),
    **regular_op_with_shaped_data('transpose', int64_array([3, 100, 66, 34, 2]),
                                 {'type': 'Transpose', 'op': 'Transpose'}),
    **regular_op_with_shaped_data('roll', int64_array([3, 100, 66, 34, 2]), {'type': 'Roll', 'op': 'Roll'}),
    **valued_const_with_data('shift', int64_array([20, 20])),
    **valued_const_with_data('axis', int64_array([1, 3, 4])),
    **regular_op_with_shaped_data('abs', int64_array([3, 100, 66, 34, 2]), {'type': 'Abs', 'op': 'Abs'}),
    **result('output'),
}

ref_graph_edges_3 = [
    *connect('placeholder:0', '0:transpose'),
    *connect('perm:0', '1:transpose'),
    *connect('transpose:0', '0:roll'),
    *connect('shift:0', '1:roll'),
    *connect('axis:0', '2:roll'),
    *connect('roll:0', '0:abs'),
    *connect('abs:0', 'output'),
]


class SSliceComplexMiddleStageTest(unittest.TestCase):
    def test_replacement_for_the_last_axis(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        SSliceComplex().find_and_replace_pattern(graph)
        graph.clean_up()
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_nonreplacement_for_the_last_axis(self):
        graph = build_graph(nodes_attrs=non_transformed_graph_node_attrs, edges=non_transformed_graph_edges)
        ref_graph = build_graph(nodes_attrs=non_transformed_graph_node_attrs, edges=non_transformed_graph_edges)
        SSliceComplex().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_replacement_for_non_last_axis(self):
        graph = build_graph(nodes_attrs=graph_node_attrs_2, edges=graph_edges)
        SSliceComplex().find_and_replace_pattern(graph)
        graph.clean_up()
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs_2, edges=ref_graph_edges_2)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_replacement_with_update_roll_axes(self):
        graph = build_graph(nodes_attrs=graph_node_attrs_3, edges=graph_edges_2)
        SSliceComplex().find_and_replace_pattern(graph)
        graph.clean_up()
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs_3, edges=ref_graph_edges_3)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
