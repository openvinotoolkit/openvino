# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.TFFFTToDFT import TFFFTToDFT
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
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


dft_graph_with_signal_size_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'fft': {'kind': 'op', 'op': 'TFFFT', 'num_of_dimensions': 2, 'is_inverse': False},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    'signal_size': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
}

dft_graph_with_signal_size_edges = [
    ('placeholder', 'fft', {'in': 0}),
    ('signal_size', 'fft', {'in': 1}),
    ('fft', 'abs'),
    ('abs', 'output'),
]


ref_dft_graph_with_signal_size_node_attrs = {
    'placeholder': {'shape': int64_array([3, 100, 100, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'fft': {'kind': 'op', 'op': 'DFT'},
    'abs': {'type': 'Abs', 'kind': 'op', 'op': 'Abs'},
    'output': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
    'fft_axes': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
    'signal_size': {
        'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': int64_array([2]), 'value': int64_array([-2, -1])
    },
}

ref_dft_graph_with_signal_size_edges = [
    ('placeholder', 'fft', {'in': 0}),
    ('fft', 'abs'),
    ('abs', 'output'),
    ('fft_axes', 'fft', {'in': 1}),
    ('signal_size', 'fft', {'in': 2}),
]


class TestTFFFTToDFTTest():
    @pytest.mark.parametrize("num_of_dimensions, dft_type, fft_axes",[(2, 'DFT', int64_array([-2, -1])),
                (2, 'IDFT', int64_array([-2, -1])),
                (1, 'DFT', int64_array([-1])),
                (1, 'IDFT', int64_array([-1])),
                (3, 'DFT', int64_array([-3, -2, -1])),
                (3, 'IDFT', int64_array([-3, -2, -1])),
                (2, 'RDFT', int64_array([-2, -1])),
                (2, 'IRDFT', int64_array([-2, -1])),
                (1, 'RDFT', int64_array([-1])),
                (1, 'IRDFT', int64_array([-1])),
                (3, 'RDFT', int64_array([-3, -2, -1])),
                (3, 'IRDFT', int64_array([-3, -2, -1]))])
    def test_replacement(self, num_of_dimensions, dft_type, fft_axes):
        graph = build_graph(nodes_attrs=dft_graph_node_attrs,
                            edges=dft_graph_edges,
                            update_attributes={
                                'fft': {'num_of_dimensions': num_of_dimensions, 'fft_kind': dft_type},
                            })
        graph.stage = 'front'
        graph.graph['layout'] = 'NHWC'
        TFFFTToDFT().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_dft_graph_node_attrs,
                                edges=ref_dft_graph_edges,
                                update_attributes={
                                    'fft': {'kind': 'op', 'op': dft_type},
                                    'fft_axes': {'value': fft_axes, 'shape': int64_array(fft_axes.shape)},
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        assert flag, resp

    @pytest.mark.parametrize("num_of_dims, fft_kind, fft_axes, input_shape, signal_size",[
        (2, 'RDFT', int64_array([-2, -1]), int64_array([3, 100, 100]), int64_array([100, -1])),
        (2, 'IRDFT', int64_array([-2, -1]), int64_array([3, 100, 100, 2]), int64_array([100, -1])),
        (2, 'RDFT', int64_array([-2, -1]), int64_array([3, 100, 100]), int64_array([95, 116])),
        (2, 'IRDFT', int64_array([-2, -1]), int64_array([3, 100, 100, 2]), int64_array([95, 116])),

        (3, 'RDFT', int64_array([-3, -2, -1]), int64_array([3, 100, 100]), int64_array([5, 100, -1])),
        (3, 'IRDFT', int64_array([-3, -2, -1]), int64_array([3, 100, 100, 2]), int64_array([5, 100, -1])),
        (3, 'RDFT', int64_array([-3, -2, -1]), int64_array([3, 100, 100]), int64_array([5, 95, 116])),
        (3, 'IRDFT', int64_array([-3, -2, -1]), int64_array([3, 100, 100, 2]), int64_array([5, 95, 116])),

        (1, 'RDFT', int64_array([-1]), int64_array([3, 100, 100]), int64_array([-1])),
        (1, 'IRDFT', int64_array([-1]), int64_array([3, 100, 100, 2]), int64_array([-1])),
        (1, 'RDFT', int64_array([-1]), int64_array([3, 100, 100]), int64_array([95])),
        (1, 'IRDFT', int64_array([-1]), int64_array([3, 100, 100, 2]), int64_array([95])),
        (1, 'RDFT', int64_array([-1]), int64_array([3, 100, 100]), int64_array([116])),
        (1, 'IRDFT', int64_array([-1]), int64_array([3, 100, 100, 2]), int64_array([116])),
        (1, 'RDFT', int64_array([-1]), int64_array([3, 100, 100]), int64_array([100])),
        (1, 'IRDFT', int64_array([-1]), int64_array([3, 100, 100, 2]), int64_array([100])),
    ])
    def test_replacement_for_signal_size(self, num_of_dims, fft_kind, fft_axes, input_shape, signal_size):
        graph = build_graph(nodes_attrs=dft_graph_with_signal_size_node_attrs,
                            edges=dft_graph_with_signal_size_edges,
                            update_attributes={
                                'placeholder': {'shape': input_shape},
                                'signal_size': {
                                    'shape': signal_size.shape, 'value': signal_size
                                },
                                'fft': {'num_of_dimensions': num_of_dims, 'fft_kind': fft_kind},
                            })
        graph.stage = 'front'
        graph.graph['layout'] = 'NHWC'
        TFFFTToDFT().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=ref_dft_graph_with_signal_size_node_attrs,
                                edges=ref_dft_graph_with_signal_size_edges,
                                update_attributes={
                                    'placeholder': {'shape': input_shape},
                                    'signal_size': {
                                        'shape': signal_size.shape, 'value': signal_size
                                    },
                                    'fft': {'kind': 'op', 'op': fft_kind},
                                    'fft_axes': {'value': fft_axes, 'shape': int64_array(fft_axes.shape)},
                                })
        (flag, resp) = compare_graphs(graph, ref_graph, 'output', check_op_attrs=True)
        assert flag, resp
