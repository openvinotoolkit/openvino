# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.RFFTRealImagToRFFTSplit import RFFTRealImagToRDFTSplit
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 192, 36, 64]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'rfft': {'kind': 'op', 'op': 'TFFFT', 'num_of_dimensions': 2, 'fft_kind': 'RDFT'},
    'real': {'kind': 'op', 'op': 'Real'},
    'imag': {'kind': 'op', 'op': 'Imag'},
    'real_sigmoid': {'type': 'Sigmoid', 'kind': 'op', 'op': 'Sigmoid'},
    'imag_sigmoid': {'type': 'Sigmoid', 'kind': 'op', 'op': 'Sigmoid'},
    'rfft_lengths': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([36, 33])
    },
    'add': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
}

graph_edges = [
    ('placeholder', 'rfft', {'in': 0}),
    ('rfft', 'real', {'out': 0, 'in': 0}),
    ('rfft', 'imag', {'out': 0, 'in': 0}),
    ('real', 'real_sigmoid', {'in': 0}),
    ('imag', 'imag_sigmoid', {'in': 0}),
    ('real_sigmoid', 'add', {'in': 0}),
    ('imag_sigmoid', 'add', {'in': 1}),
    ('rfft_lengths', 'rfft', {'in': 1}),
    ('add', 'abs'),
    ('abs', 'output'),
]


ref_graph_node_attrs = {
    'placeholder': {'shape': int64_array([3, 192, 36, 64]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'rfft': {'kind': 'op', 'op': 'TFFFT', 'num_of_dimensions': 2, 'fft_kind': 'RDFT'},
    'real': {'kind': 'op', 'op': 'Real'},
    'imag': {'kind': 'op', 'op': 'Imag'},
    'real_sigmoid': {'type': 'Sigmoid', 'kind': 'op', 'op': 'Sigmoid'},
    'imag_sigmoid': {'type': 'Sigmoid', 'kind': 'op', 'op': 'Sigmoid'},
    'rfft_lengths': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([36, 33])
    },
    'add': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 2},
    'split_axis': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array(-1).shape, 'value': int64_array(-1)
    },
    'real_squeeze': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    'imag_squeeze': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
    'real_squeeze_axis': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array(-1).shape, 'value': int64_array(-1)
    },
    'imag_squeeze_axis': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array(-1).shape, 'value': int64_array(-1)
    },
}

ref_graph_edges = [
    ('placeholder', 'rfft', {'in': 0}),
    ('rfft', 'split', {'in': 0, 'out': 0}),
    ('split_axis', 'split', {'in': 1}),
    ('split', 'real_squeeze', {'in': 0, 'out': 0}),
    ('split', 'imag_squeeze', {'in': 0, 'out': 1}),
    ('real_squeeze_axis', 'real_squeeze', {'in': 1}),
    ('imag_squeeze_axis', 'imag_squeeze', {'in': 1}),
    ('rfft_lengths', 'rfft', {'in': 1}),
    ('real_squeeze', 'real_sigmoid', {'in': 0}),
    ('imag_squeeze', 'imag_sigmoid', {'in': 0}),
    ('real_sigmoid', 'add', {'in': 0}),
    ('imag_sigmoid', 'add', {'in': 1}),
    ('add', 'abs'),
    ('abs', 'output'),
]


class TestRFFTRealImagToRFFTSplitTest():
    @pytest.mark.parametrize("num_of_dims",[1, 2, 3])
    def test_replacement(self, num_of_dims):
        graph = build_graph(nodes_attrs=graph_node_attrs,
                            edges=graph_edges,
                            update_attributes={
                                'rfft': {'num_of_dimensions': num_of_dims}
                            })
        graph.stage = 'front'
        RFFTRealImagToRDFTSplit().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_graph_node_attrs,
                                edges=ref_graph_edges,
                                update_attributes={
                                    'rfft': {'num_of_dimensions': num_of_dims}
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        assert flag, resp
