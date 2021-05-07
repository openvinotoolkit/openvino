# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

from generator import generator, generate

from extensions.front.tf.TFFFTToDFT import TFFFTToDFT
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


dft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'fft': {'kind': 'op', 'op': 'TFFFT', 'num_of_dimensions': 2, 'is_inverse': False},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

dft_graph_edges = [
    ('placeholder', 'fft', {'in': 0}),
    ('fft', 'abs'),
    ('abs', 'output'),
]


ref_dft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'fft': {'kind': 'op', 'op': 'DFT'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    'fft_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
}

ref_dft_graph_edges = [
    ('placeholder', 'fft', {'in': 0}),
    ('fft', 'abs'),
    ('abs', 'output'),
    ('fft_axes', 'fft', {'in': 1}),
]


@generator
class TFFFTToDFTTest(unittest.TestCase):
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
        graph.graph['layout'] = 'NHWC'
        TFFFTToDFT().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_dft_graph_node_attrs,
                                edges=ref_dft_graph_edges,
                                update_attributes={
                                    'fft': {'kind': 'op', 'op': dft_type},
                                    'fft_axes': {'value': fft_axes, 'shape': int64_array(fft_axes.shape)},
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
