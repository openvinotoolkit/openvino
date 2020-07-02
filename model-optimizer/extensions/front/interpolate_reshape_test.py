"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import unittest
from argparse import Namespace

import numpy as np
from generator import generator, generate

from extensions.front.interpolate_reshape import InterpolateWithConcat
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, regular_op_with_shaped_data, valued_const_with_data, connect, \
    connect_data

nodes = {
    **regular_op_with_shaped_data('placeholder', [1, 3, 30, 40], {'type': 'Parameter', 'op': 'Parameter'}),
    **valued_const_with_data('out_shape', np.array([60, 160])),

    **regular_op_with_shaped_data('interpolate', [1, 3, 60, 160],
                                  {'type': 'Interpolate', 'axes': int64_array([2, 3]), 'op': 'Interpolate'}),
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

    **result(),
}


@generator
class TestInterpolateConcat(unittest.TestCase):
    def test_interpolate_concat_reshape_graph_comparison(self):
        graph = build_graph(nodes, [
            *connect('placeholder', '0:interpolate'),
            *connect('out_shape', '1:interpolate'),
            *connect('interpolate', '0:concat'),
            *connect('placeholder_1', '1:concat'),
            *connect('concat', 'output'),
        ], nodes_with_edges_only=True)

        InterpolateWithConcat().find_and_replace_pattern(graph)
        graph.graph['cmd_params'] = Namespace(keep_shape_ops=True)
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
        self.assertTrue(flag, resp)

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
        graph.graph['cmd_params'] = Namespace(keep_shape_ops=True)
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
        self.assertTrue(flag, resp)

    @generate(*[
        {'concat': {'axis': None}},
        {'concat': {'axis': 2}},
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
        self.assertTrue(flag, resp)
