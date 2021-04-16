# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

from generator import generator, generate

from extensions.front.tf.SSliceComplexRolledFFTPackBlockReplacement import SSliceComplexRolledFFTPackBlockReplacement
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


dft_graph_node_attrs = {
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
    'roll': {'kind': 'op', 'op': 'TFRoll'},
    'fft': {'kind': 'op', 'op': 'TFFFT', 'num_of_dimensions': 2, 'is_inverse': False},
    'unroll': {'kind': 'op', 'op': 'TFRoll'},
    'real': {'kind': 'op', 'op': 'Real'},
    'imag': {'kind': 'op', 'op': 'Imag'},
    'pack': {'kind': 'op', 'op': 'Pack'},
    'roll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'roll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
    'unroll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'unroll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
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

dft_graph_edges = [
    ('placeholder', 'strided_slice_real', {'out': 0, 'in': 0}),
    ('placeholder', 'strided_slice_imag', {'out': 0, 'in': 0}),
    ('strided_slice_real', 'complex', {'in': 0}),
    ('strided_slice_imag', 'complex', {'in': 1}),
    ('complex', 'roll', {'in': 0}),
    ('roll', 'fft', {'in': 0}),
    ('fft', 'unroll', {'in': 0}),
    ('unroll', 'real', {'out': 0, 'in': 0}),
    ('unroll', 'imag', {'out': 0, 'in': 0}),
    ('real', 'pack', {'in': 0}),
    ('imag', 'pack', {'in': 1}),
    ('pack', 'abs'),
    ('abs', 'output'),
    ('roll_shift', 'roll', {'in': 1}),
    ('roll_axes', 'roll', {'in': 2}),
    ('unroll_shift', 'unroll', {'in': 1}),
    ('unroll_axes', 'unroll', {'in': 2}),
    ('real_begin', 'strided_slice_real', {'in': 1}),
    ('imag_begin', 'strided_slice_imag', {'in': 1}),
    ('real_end', 'strided_slice_real', {'in': 2}),
    ('imag_end', 'strided_slice_imag', {'in': 2}),
    ('real_strides', 'strided_slice_real', {'in': 3}),
    ('imag_strides', 'strided_slice_imag', {'in': 3}),
]


ref_dft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'roll': {'kind': 'op', 'op': 'TFRoll'},
    'fft': {'kind': 'op', 'op': 'DFT'},
    'unroll': {'kind': 'op', 'op': 'TFRoll'},
    'roll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'roll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
    'unroll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'unroll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    'fft_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
}

ref_dft_graph_edges = [
    ('placeholder', 'roll', {'out': 0, 'in': 0}),
    ('roll', 'fft', {'in': 0}),
    ('fft', 'unroll', {'in': 0}),
    ('unroll', 'abs', {'out': 0, 'in': 0}),
    ('abs', 'output'),
    ('roll_shift', 'roll', {'in': 1}),
    ('unroll_shift', 'unroll', {'in': 1}),
    ('roll_axes', 'roll', {'in': 2}),
    ('unroll_axes', 'unroll', {'in': 2}),
    ('fft_axes', 'fft', {'in': 1}),
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
    'roll': {'kind': 'op', 'op': 'TFRoll'},
    'fft': {'kind': 'op', 'op': 'TFFFT', 'num_of_dimensions': 2, 'is_inverse': False},
    'unroll': {'kind': 'op', 'op': 'TFRoll'},
    'real': {'kind': 'op', 'op': 'Real'},
    'imag': {'kind': 'op', 'op': 'Imag'},
    'pack': {'kind': 'op', 'op': 'Pack'},
    'roll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'roll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
    'unroll_shift': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([50, 50])
    },
    'unroll_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
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
    ('complex', 'roll', {'in': 0}),
    ('roll', 'fft', {'in': 0}),
    ('fft', 'unroll', {'in': 0}),
    ('unroll', 'real', {'out': 0, 'in': 0}),
    ('unroll', 'imag', {'out': 0, 'in': 0}),
    ('real', 'pack', {'in': 0}),
    ('imag', 'pack', {'in': 1}),
    ('pack', 'abs'),
    ('abs', 'output'),
    ('roll_shift', 'roll', {'in': 1}),
    ('roll_axes', 'roll', {'in': 2}),
    ('unroll_shift', 'unroll', {'in': 1}),
    ('unroll_axes', 'unroll', {'in': 2}),
    ('real_begin', 'strided_slice_real', {'in': 1}),
    ('imag_begin', 'strided_slice_imag', {'in': 1}),
    ('real_end', 'strided_slice_real', {'in': 2}),
    ('imag_end', 'strided_slice_imag', {'in': 2}),
    ('real_strides', 'strided_slice_real', {'in': 3}),
    ('imag_strides', 'strided_slice_imag', {'in': 3}),
]


@generator
class SSliceComplexRolledFFTPackBlockReplacementTest(unittest.TestCase):
    @generate(*[(2, False, 'DFT', int64_array([-2, -1])),
                (2, True, 'IDFT', int64_array([-2, -1])),
                (1, False, 'DFT', int64_array([-1])),
                (1, True, 'IDFT', int64_array([-1])),
                (3, False, 'DFT', int64_array([-3, -2, -1])),
                (3, True, 'IDFT', int64_array([-3, -2, -1]))])
    def test_replacement(self, num_of_dimensions, is_inverse, dft_type, fft_axes):
        graph = build_graph(nodes_attrs=dft_graph_node_attrs,
                            edges=dft_graph_edges,
                            update_attributes={
                                'fft': {'num_of_dimensions': num_of_dimensions, 'is_inverse': is_inverse},
                            })
        graph.stage = 'front'
        setattr(graph.graph['cmd_params'], 'disable_nhwc_to_nchw', False)
        SSliceComplexRolledFFTPackBlockReplacement().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_dft_graph_node_attrs,
                                edges=ref_dft_graph_edges,
                                update_attributes={
                                    'fft': {'kind': 'op', 'op': dft_type},
                                    'fft_axes': {'value': fft_axes, 'shape': int64_array(fft_axes.shape)},
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    @generate(*[(2, False), (2, True), (1, False), (1, True), (3, False), (3, True)])
    def test_not_replacement(self, num_of_dimensions, is_inverse):
        graph = build_graph(nodes_attrs=non_transformed_graph_node_attrs,
                            edges=non_transformed_graph_edges,
                            update_attributes={
                                'fft': {'num_of_dimensions': num_of_dimensions, 'is_inverse': is_inverse},
                            })
        ref_graph = build_graph(nodes_attrs=non_transformed_graph_node_attrs,
                                edges=non_transformed_graph_edges,
                                update_attributes={
                                    'fft': {'num_of_dimensions': num_of_dimensions, 'is_inverse': is_inverse},
                                })
        graph.stage = 'front'
        setattr(graph.graph['cmd_params'], 'disable_nhwc_to_nchw', False)
        SSliceComplexRolledFFTPackBlockReplacement().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
