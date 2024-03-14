# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from openvino.tools.mo.front.interpolate_reshape import InterpolateWithConcat
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect, \
    connect_data

nodes = {
    **regular_op_with_shaped_data('placeholder', [1, 3, 30, 40], {'type': 'Parameter', 'op': 'Parameter'}),
    **valued_const_with_data('out_shape', np.array([60, 160])),

    **regular_op_with_shaped_data('interpolate', [1, 3, 60, 160],
                                  {'type': 'Interpolate', 'axes': int64_array([2, 3]), 'op': 'Interpolate',
                                   'version': 'opset1'}),
    **regular_op_with_shaped_data('identity_00', [1, 3, 60, 160], {'identity': True, 'op': 'Identity'}),
    **regular_op_with_shaped_data('identity_01', [1, 3, 60, 160], {'identity': True, 'op': 'Identity'}),

    **regular_op_with_shaped_data('shape', [4], {'type': 'ShapeOf', 'op': 'ShapeOf'}),
    **valued_const_with_data('indices', np.array([2, 3])),
    **valued_const_with_data('axis', np.array(0)),
    **regular_op_with_shaped_data('gather', [2], {'type': 'Gather', 'op': 'Gather'}),

    **regular_op_with_shaped_data('placeholder_1', [1, 3, 60, 160], {'type': 'Parameter', 'op': 'Parameter'}),
    **regular_op_with_shaped_data('identity_10', [1, 3, 60, 160], {'identity': True, 'op': 'Identity'}),
    **regular_op_with_shaped_data('identity_11', [1, 3, 60, 160], {'identity': True, 'op': 'Identity'}),
    **regular_op_with_shaped_data('concat', [1, 7, 60, 160], {'type': 'Concat', 'axis': 1, 'op': 'Concat'}),

    **valued_const_with_data('N', np.array([1])),

    **result('output'),
    **result('output_1'),
}


class TestInterpolateConcat():
    def test_interpolate_concat_reshape_graph_comparison(self):
        graph = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', '0:concat'),
            *connect('placeholder_1', '1:concat'),
            *connect('concat', 'output'),
        ], nodes_with_edges_only=True)

        InterpolateWithConcat().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('placeholder_1', 'shape'),
            *connect('shape', '0:gather'),
            *connect('indices', '1:gather'),
            *connect('axis', '2:gather'),
            *connect('gather', '1:interpolate'),
            *connect('interpolate', '0:concat'),
            *connect_data('placeholder_1', '1:concat'),
            *connect('concat', 'output'),
        ], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

    def test_interpolate_identity_concat_reshape_graph_comparison(self):
        graph = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', 'identity_00'),
            *connect('identity_00', 'identity_01'),
            *connect('identity_01', '0:concat'),
            *connect('placeholder_1', 'identity_10'),
            *connect('identity_10', 'identity_11'),
            *connect('identity_11', '1:concat'),
            *connect('concat', 'output'),
        ], nodes_with_edges_only=True)

        InterpolateWithConcat().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect_data('identity_11', 'shape'),
            *connect('shape', '0:gather'),
            *connect('indices', '1:gather'),
            *connect('axis', '2:gather'),
            *connect('gather', '1:interpolate'),
            *connect('interpolate', 'identity_00'),
            *connect('identity_00', 'identity_01'),
            *connect('identity_01', '0:concat'),
            *connect('placeholder_1', 'identity_10'),
            *connect('identity_10', 'identity_11'),
            *connect('identity_11', '1:concat'),
            *connect('concat', 'output'),
        ], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

    def test_interpolate_concat_negate(self):
        graph = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', 'identity_00'),
            *connect('interpolate', 'identity_01'),
            *connect('identity_00', 'output'),
            *connect('identity_01', 'output_1'),
        ], nodes_with_edges_only=True)

        InterpolateWithConcat().find_and_replace_pattern(graph)
        graph.clean_up()
        graph_ref = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', 'identity_00'),
            *connect('interpolate', 'identity_01'),
            *connect('identity_00', 'output'),
            *connect('identity_01', 'output_1'),
        ], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

    @pytest.mark.parametrize("update_attrs",[
        {'concat': {'axis': None}},

        {'concat': {'axis': -1}},
        {'interpolate': {'axes': None}},
        {'interpolate': {'axes': np.array([1])}},
        {'interpolate': {'axes': np.array([2, -1])}},
    ])
    def test_negative_axes_conditions(self, update_attrs):
        graph = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', '0:concat'),
            *connect('placeholder_1', '1:concat'),
            *connect('concat', 'output'),
        ], update_attributes=update_attrs, nodes_with_edges_only=True)
        InterpolateWithConcat().find_and_replace_pattern(graph)
        graph_ref = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', '0:concat'),
            *connect('placeholder_1', '1:concat'),
            *connect('concat', 'output'),
        ], update_attributes=update_attrs, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

    def test_interpolate_tf_style_concat(self):
        graph = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', '0:concat'),
            *connect('N', '1:concat'),
            *connect('concat', 'output'),
        ], update_attributes={'concat': {'N': 1}}, nodes_with_edges_only=True)
        graph_ref = graph.copy()
        InterpolateWithConcat().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp
