# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

from extensions.front.tf.SSliceComplex import SSliceComplex
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'strided_slice_real': {
        'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
        'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
        'shrink_axis_mask': int64_array([0, 1]),
    },
    'strided_slice_imag': {
        'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
        'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
        'shrink_axis_mask': int64_array([0, 1]),
    },
    'complex': {'kind': 'op', 'op': 'Complex'},
    'real_begin': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 0])
    },
    'imag_begin': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 1])
    },
    'real_end': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 1])
    },
    'imag_end': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 2])
    },
    'real_strides': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([1, 1])
    },
    'imag_strides': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([1, 1])
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

graph_edges = [
    ('placeholder', 'strided_slice_real', {'out': 0, 'in': 0}),
    ('placeholder', 'strided_slice_imag', {'out': 0, 'in': 0}),
    ('strided_slice_real', 'complex', {'in': 0}),
    ('strided_slice_imag', 'complex', {'in': 1}),
    ('complex', 'abs'),
    ('abs', 'output'),
    ('real_begin', 'strided_slice_real', {'in': 1}),
    ('imag_begin', 'strided_slice_imag', {'in': 1}),
    ('real_end', 'strided_slice_real', {'in': 2}),
    ('imag_end', 'strided_slice_imag', {'in': 2}),
    ('real_strides', 'strided_slice_real', {'in': 3}),
    ('imag_strides', 'strided_slice_imag', {'in': 3}),
]


ref_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

ref_graph_edges = [
    ('placeholder', 'abs'),
    ('abs', 'output'),
]


non_transformed_graph_node_attrs = {
    'placeholder_0': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'strided_slice_real': {
        'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
        'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
        'shrink_axis_mask': int64_array([0, 1]),
    },
    'strided_slice_imag': {
        'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice', 'begin_mask': int64_array([1]),
        'end_mask': int64_array([1]), 'ellipsis_mask': int64_array([1]), 'new_axis_mask': int64_array([0]),
        'shrink_axis_mask': int64_array([0, 1]),
    },
    'complex': {'kind': 'op', 'op': 'Complex'},
    'real_begin': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 0])
    },
    'imag_begin': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 1])
    },
    'real_end': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 1])
    },
    'imag_end': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([0, 2])
    },
    'real_strides': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([1, 1])
    },
    'imag_strides': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([1, 1])
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

non_transformed_graph_edges = [
    ('placeholder_0', 'strided_slice_real', {'out': 0, 'in': 0}),
    ('placeholder_1', 'strided_slice_imag', {'out': 0, 'in': 0}),
    ('strided_slice_real', 'complex', {'in': 0}),
    ('strided_slice_imag', 'complex', {'in': 1}),
    ('complex', 'abs'),
    ('abs', 'output'),
    ('real_begin', 'strided_slice_real', {'in': 1}),
    ('imag_begin', 'strided_slice_imag', {'in': 1}),
    ('real_end', 'strided_slice_real', {'in': 2}),
    ('imag_end', 'strided_slice_imag', {'in': 2}),
    ('real_strides', 'strided_slice_real', {'in': 3}),
    ('imag_strides', 'strided_slice_imag', {'in': 3}),
]


class SSliceComplexTest(unittest.TestCase):
    def test_replacement(self):
        graph = build_graph(nodes_attrs=graph_node_attrs, edges=graph_edges)
        graph.stage = 'front'
        SSliceComplex().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs, edges=ref_graph_edges)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_nonreplacement(self):
        graph = build_graph(nodes_attrs=non_transformed_graph_node_attrs, edges=non_transformed_graph_edges)
        ref_graph = build_graph(nodes_attrs=non_transformed_graph_node_attrs, edges=non_transformed_graph_edges)
        graph.stage = 'front'
        SSliceComplex().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
