# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import unittest

from generator import generator, generate

from extensions.front.mxnet.MXFFTToDFT import MXFFTToDFT
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


fft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'fft': {'kind': 'op', 'op': 'MXFFT', 'is_inverse': False},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

fft_graph_edges = [
    ('placeholder', 'fft', {'in': 0}),
    ('fft', 'abs'),
    ('abs', 'output'),
]


ref_converted_fft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'rank': {'kind': 'op', 'op': 'Rank'},
    'unsqueeze': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
    'unsqueeze_axis': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([1]), 'value': int64_array([-1])
    },
    'one': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(1)
    },
    'add': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'zero1': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(0)
    },
    'broadcast1': {'type': 'Broadcast', 'kind': 'op', 'op': 'Broadcast'},
    'one2': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(1)
    },
    'zero2': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(0)
    },
    'scatter': {'type': 'ScatterUpdate', 'kind': 'op', 'op': 'ScatterUpdate'},
    'pad': {'type': 'Pad', 'kind': 'op', 'op': 'Pad', 'mode': 'constant'},
    'fft_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([1]), 'value': int64_array([-1])
    },
    'fft': {'kind': 'op', 'op': 'DFT', 'type': 'DFT'},
    'one3': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(1)
    },
    'sub': {'type': 'Subtract', 'kind': 'op', 'op': 'Sub'},
    'zero3': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(0)
    },
    'broadcast2': {'type': 'Broadcast', 'kind': 'op', 'op': 'Broadcast'},
    'm1_2': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-1, 2])
    },
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat', 'axis': 0},
    'reshape': {'kind': 'op', 'op': 'Reshape', 'type': 'Reshape'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

ref_converted_fft_graph_edges = [
    ('placeholder', 'rank', {'in': 0, 'out': 0}),
    ('placeholder', 'unsqueeze', {'in': 0, 'out': 0}),
    ('unsqueeze_axis', 'unsqueeze', {'in': 1, 'out': 0}),
    ('rank', 'add', {'in': 0, 'out': 0}),
    ('one', 'add', {'in': 1, 'out': 0}),
    ('zero1', 'broadcast1', {'in': 0, 'out': 0}),
    ('add', 'broadcast1', {'in': 1, 'out': 0}),
    ('broadcast1', 'scatter', {'in': 0, 'out': 0}),
    ('rank', 'scatter', {'in': 1, 'out': 0}),
    ('one2', 'scatter', {'in': 2, 'out': 0}),
    ('zero2', 'scatter', {'in': 3, 'out': 0}),
    ('unsqueeze', 'pad', {'in': 0, 'out': 0}),
    ('broadcast1', 'pad', {'in': 1, 'out': 0}),
    ('scatter', 'pad', {'in': 2, 'out': 0}),
    ('pad', 'fft', {'in': 0, 'out': 0}),
    ('fft_axes', 'fft', {'in': 1, 'out': 0}),
    ('rank', 'sub', {'in': 0, 'out': 0}),
    ('one3', 'sub', {'in': 1, 'out': 0}),
    ('zero3', 'broadcast2', {'in': 0, 'out': 0}),
    ('sub', 'broadcast2', {'in': 1, 'out': 0}),
    ('broadcast2', 'concat', {'in': 0, 'out': 0}),
    ('m1_2', 'concat', {'in': 1, 'out': 0}),
    ('fft', 'reshape', {'in': 0, 'out': 0}),
    ('concat', 'reshape', {'in': 1, 'out': 0}),
    ('reshape', 'abs'),
    ('abs', 'output'),
]


ref_converted_ifft_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'rank': {'kind': 'op', 'op': 'Rank'},
    'subtracted_one': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(1)
    },
    'sub': {'type': 'Subtract', 'kind': 'op', 'op': 'Sub'},
    'broadcast': {'type': 'Broadcast', 'kind': 'op', 'op': 'Broadcast'},
    'broadcasted_value': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(0)
    },
    'new_shape': {'type': 'Concat', 'kind': 'op', 'op': 'Concat', 'axis': 0},
    'new_shape_const': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-1, 2])
    },
    'reshape': {'kind': 'op', 'op': 'Reshape', 'type': 'Reshape'},
    'fft': {'kind': 'op', 'op': 'IDFT', 'type': 'IDFT'},
    'fft_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([1]), 'value': int64_array([-1])
    },
    'split': {'kind': 'op', 'op': 'Split', 'type': 'Split', 'num_splits': 2},
    'split_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([]), 'value': int64_array(-1)
    },
    'squeeze': {'kind': 'op', 'op': 'Squeeze', 'type': 'Squeeze'},
    'squeeze_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([1]), 'value': int64_array([-1])
    },
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

ref_converted_ifft_graph_edges = [
    ('placeholder', 'rank', {'out': 0}),
    ('placeholder', 'reshape', {'out': 0}),
    ('rank', 'sub'),
    ('subtracted_one', 'sub'),
    ('broadcasted_value', 'broadcast'),
    ('sub', 'broadcast'),
    ('broadcast', 'new_shape'),
    ('new_shape_const', 'new_shape'),
    ('new_shape', 'reshape'),
    ('reshape', 'fft'),
    ('fft_axes', 'fft'),
    ('fft', 'split'),
    ('split_axes', 'split'),
    ('split', 'squeeze', {'out': 0}),
    ('squeeze_axes', 'squeeze'),
    ('squeeze', 'abs'),
    ('abs', 'output'),
]


@generator
class MXFFTToDFTTest(unittest.TestCase):
    @generate(*[int64_array([3, 100, 100, 8]), int64_array([5, 60])])
    def test_fft_replacement(self, input_shape):
        graph = build_graph(nodes_attrs=fft_graph_node_attrs,
                            edges=fft_graph_edges,
                            update_attributes={
                                'placeholder': {'shape': input_shape}
                            })
        graph.stage = 'front'
        MXFFTToDFT().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_converted_fft_graph_node_attrs,
                                edges=ref_converted_fft_graph_edges,
                                update_attributes={
                                    'placeholder': {'shape': input_shape}
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)

    @generate(*[int64_array([3, 100, 100, 8]), int64_array([5, 60])])
    def test_ifft_replacement(self, input_shape):
        graph = build_graph(nodes_attrs=fft_graph_node_attrs,
                            edges=fft_graph_edges,
                            update_attributes={
                                'placeholder': {'shape': input_shape},
                                'fft': {'is_inverse': True}
                            })
        graph.stage = 'front'
        MXFFTToDFT().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_converted_ifft_graph_node_attrs,
                                edges=ref_converted_ifft_graph_edges,
                                update_attributes={
                                    'placeholder': {'shape': input_shape}
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
