# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging as log
import unittest

from extensions.front.tf.SSliceComplexRolledFFTPackBlockReplacement import SSliceComplexRolledFFTPackBlockReplacement
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const


dft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'strided_slice_real': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice'},
    'strided_slice_imag': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice'},
     **const('real_begin', int64_array([0, 0])),
     **const('real_end', int64_array([0, 1])),
     **const('real_strides', int64_array([1, 1])),
     **const('imag_begin', int64_array([0, 1])),
     **const('imag_end', int64_array([0, 2])),
     **const('imag_strides', int64_array([1, 1])),
    'tf_fft': {'kind': 'op', 'op': 'TFFFT', 'num_of_dimensions': 2, 'is_inverse': False},
    'complex': {'kind': 'op', 'op': 'Complex'},
    'roll': {'kind': 'op', 'op': 'TFRoll'},
    'unroll': {'kind': 'op', 'op': 'TFRoll'},
    'real': {'kind': 'op', 'op': 'Real'},
    'imag': {'kind': 'op', 'op': 'Imag'},
    'pack': {'type': 'Pack', 'kind': 'op', 'op': 'Pack'},
    **const('roll_shift', int64_array([50, 50])),
    **const('roll_axes', int64_array([-2, -1])),
    **const('unroll_shift', int64_array([50, 50])),
    **const('unroll_axes', int64_array([-2, -1])),
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

dft_graph_edges = [
    ('placeholder', 'strided_slice_real', {'out': 0, 'in': 0}),
    ('placeholder', 'strided_slice_imag', {'out': 0, 'in': 0}),
    ('real_begin', 'strided_slice_real', {'in': 1}),
    ('imag_begin', 'strided_slice_imag', {'in': 1}),
    ('real_end', 'strided_slice_real', {'in': 2}),
    ('imag_end', 'strided_slice_imag', {'in': 2}),
    ('real_strides', 'strided_slice_real', {'in': 3}),
    ('imag_strides', 'strided_slice_imag', {'in': 3}),
    ('strided_slice_real', 'complex', {'in': 0}),
    ('strided_slice_imag', 'complex', {'in': 0}),
    ('roll_shift', 'roll', {'in': 1}),
    ('unroll_shift', 'unroll', {'in': 1}),
    ('roll_axes', 'roll', {'in': 2}),
    ('unroll_axes', 'unroll', {'in': 2}),
    ('complex', 'roll', {'in': 0}),
    ('roll', 'tf_fft'),
    ('tf_fft', 'unroll', {'in': 0}),
    ('unroll', 'real', {'out': 0}),
    ('unroll', 'imag', {'out': 0}),
    ('real', 'pack'),
    ('imag', 'pack'),
    ('pack', 'abs'),
    ('abs', 'output'),
]


ref_dft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    **const('fft_axes', int64_array([-2, -1])),
    'fft': {'kind': 'op', 'op': 'DFT'},
    'roll': {'kind': 'op', 'op': 'TFRoll'},
    'unroll': {'kind': 'op', 'op': 'TFRoll'},
    **const('roll_shift', int64_array([50, 50])),
    **const('roll_axes', int64_array([-2, -1])),
    **const('unroll_shift', int64_array([50, 50])),
    **const('unroll_axes', int64_array([-2, -1])),
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

ref_dft_graph_edges = [
    ('placeholder', 'roll', {'in': 0}),
    ('roll_shift', 'roll', {'in': 1}),
    ('unroll_shift', 'unroll', {'in': 1}),
    ('roll_axes', 'roll', {'in': 2}),
    ('unroll_axes', 'unroll', {'in': 2}),
    ('roll', 'fft', {'in': 0}),
    ('fft_axes', 'fft', {'in': 1}),
    ('fft', 'unroll', {'in': 0}),
    ('unroll', 'abs'),
    ('abs', 'output'),
]


class SSliceComplexRolledFFTPackBlockReplacementTest(unittest.TestCase):
    def test_replacement(self):
        graph = build_graph(nodes_attrs=dft_graph_node_attrs, edges=dft_graph_edges)
        ref_graph = build_graph(nodes_attrs=ref_dft_graph_node_attrs, edges=ref_dft_graph_edges)
        # SSliceComplexRolledFFTPackBlockReplacement().replace_sub_graph(graph)
        # (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        # self.assertTrue(flag, resp)
