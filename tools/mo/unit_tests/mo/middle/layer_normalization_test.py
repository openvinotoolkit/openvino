# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.layer_normalization import LayerNormalization
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, shaped_parameter, regular_op_with_empty_data, shaped_const_with_data, \
    result, connect


class LayerNormalizationTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **regular_op_with_empty_data('layer_norm', {'op': 'LayerNorm', 'epsilon': 0.001, 'axis': -1,
                                                            'output_mean_var': False}),
                **shaped_const_with_data('gamma', None),
                **shaped_const_with_data('beta', None),
                **result('result')
            },
            edges=[
                *connect('input', '0:layer_norm'),
                *connect('gamma', '1:layer_norm'),
                *connect('beta', '2:layer_norm'),
                *connect('layer_norm', 'result')
            ]
        )

        ref_graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **shaped_const_with_data('mvn_const', None),
                **regular_op_with_empty_data('mvn', {'eps': 0.001, 'across_channels': 1, 'normalize_variance': 1,
                                                     'eps_mode': 'inside_sqrt', 'op': 'MVN', 'type': 'MVN'}),
                **shaped_const_with_data('gamma', None),
                **regular_op_with_empty_data('gamma_unsqueeze', {'op': 'Unsqueeze', 'type': 'Unsqueeze'}),
                **shaped_const_with_data('gamma_unsqueeze_const', None),
                **regular_op_with_empty_data('beta_unsqueeze', {'op': 'Unsqueeze', 'type': 'Unsqueeze'}),
                **shaped_const_with_data('beta_unsqueeze_const', None),
                **regular_op_with_empty_data('mul', {'op': 'Mul', 'type': 'Multiply'}),
                **shaped_const_with_data('beta', None),
                **regular_op_with_empty_data('add', {'op': 'Add', 'type': 'Add'}),
                **result('result')
            },
            edges=[
                *connect('input', '0:mvn'),
                *connect('mvn_const', '1:mvn'),
                *connect('mvn', '0:mul'),
                *connect('gamma', 'gamma_unsqueeze'),
                *connect('gamma_unsqueeze_const', '1:gamma_unsqueeze'),
                *connect('gamma_unsqueeze', '1:mul'),
                *connect('mul', '0:add'),
                *connect('beta', 'beta_unsqueeze'),
                *connect('beta_unsqueeze_const', '1:beta_unsqueeze'),
                *connect('beta_unsqueeze', '1:add'),
                *connect('add', 'result')
            ],
            update_attributes={
                'mvn_const': {'value': int64_array([-1]), 'shape': int64_array([1])},
                'gamma_unsqueeze_const': {'value': int64_array([0, 1, 2]), 'shape': int64_array([3])},
                'beta_unsqueeze_const': {'value': int64_array([0, 1, 2]), 'shape': int64_array([3])}
            }
        )
        LayerNormalization().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **regular_op_with_empty_data('layer_norm', {'op': 'LayerNorm', 'epsilon': 0.001, 'axis': 1,
                                                            'output_mean_var': False}),
                **shaped_const_with_data('gamma', None),
                **shaped_const_with_data('beta', None),
                **result('result')
            },
            edges=[
                *connect('input', '0:layer_norm'),
                *connect('gamma', '1:layer_norm'),
                *connect('beta', '2:layer_norm'),
                *connect('layer_norm', 'result')
            ]
        )

        ref_graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **shaped_const_with_data('mvn_const', None),
                **regular_op_with_empty_data('mvn', {'eps': 0.001, 'across_channels': 1, 'normalize_variance': 1,
                                                     'eps_mode': 'inside_sqrt', 'op': 'MVN', 'type': 'MVN'}),
                **shaped_const_with_data('gamma', None),
                **regular_op_with_empty_data('gamma_unsqueeze', {'op': 'Unsqueeze', 'type': 'Unsqueeze'}),
                **shaped_const_with_data('gamma_unsqueeze_const', None),
                **regular_op_with_empty_data('beta_unsqueeze', {'op': 'Unsqueeze', 'type': 'Unsqueeze'}),
                **shaped_const_with_data('beta_unsqueeze_const', None),
                **regular_op_with_empty_data('mul', {'op': 'Mul', 'type': 'Multiply'}),
                **shaped_const_with_data('beta', None),
                **regular_op_with_empty_data('add', {'op': 'Add', 'type': 'Add'}),
                **result('result')
            },
            edges=[
                *connect('input', '0:mvn'),
                *connect('mvn_const', '1:mvn'),
                *connect('mvn', '0:mul'),
                *connect('gamma', 'gamma_unsqueeze'),
                *connect('gamma_unsqueeze_const', '1:gamma_unsqueeze'),
                *connect('gamma_unsqueeze', '1:mul'),
                *connect('mul', '0:add'),
                *connect('beta', 'beta_unsqueeze'),
                *connect('beta_unsqueeze_const', '1:beta_unsqueeze'),
                *connect('beta_unsqueeze', '1:add'),
                *connect('add', 'result')
            ],
            update_attributes={
                'mvn_const': {'value': int64_array([1]), 'shape': int64_array([1])},
                'gamma_unsqueeze_const': {'value': int64_array([0, 2, 3]), 'shape': int64_array([3])},
                'beta_unsqueeze_const': {'value': int64_array([0, 2, 3]), 'shape': int64_array([3])}
            }
        )
        LayerNormalization().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **regular_op_with_empty_data('layer_norm', {'op': 'LayerNorm', 'epsilon': 0.001, 'axis': -1,
                                                            'output_mean_var': True}),
                **shaped_const_with_data('gamma', None),
                **shaped_const_with_data('beta', None),
                **result('result'),
                **result('result_1'),
                **result('result_2')
            },
            edges=[
                *connect('input', '0:layer_norm'),
                *connect('gamma', '1:layer_norm'),
                *connect('beta', '2:layer_norm'),
                *connect('layer_norm:0', 'result'),
                *connect('layer_norm:1', 'result_1'),
                *connect('layer_norm:2', 'result_2')
            ]
        )

        with self.assertRaises(Error):
            LayerNormalization().find_and_replace_pattern(graph)
