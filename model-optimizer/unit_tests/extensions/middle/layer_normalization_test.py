# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.middle.layer_normalization import LayerNormalization
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, shaped_parameter, regular_op_with_empty_data, result, connect, \
    shaped_const_with_data


class LayerNormalizationTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **regular_op_with_empty_data('layer_norm', {'op': 'LayerNorm', 'epsilon': 0.001, 'axis': -1}),
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
                **shaped_const_with_data('squeeze_const', None),
                **regular_op_with_empty_data('squeeze', {'op': 'Squeeze', 'type': 'Squeeze'}),
                **shaped_const_with_data('mvn_const', None),
                **regular_op_with_empty_data('mvn', {'eps': 0.001, 'across_channels': 1, 'normalize_variance': 1,
                                                     'eps_mode': 'inside_sqrt', 'op': 'MVN', 'type': 'MVN'}),
                **shaped_const_with_data('unsqueeze_const', None),
                **regular_op_with_empty_data('unsqueeze', {'op': 'Unsqueeze', 'type': 'Unsqueeze'}),
                **shaped_const_with_data('gamma', None),
                **regular_op_with_empty_data('mul', {'op': 'Mul', 'type': 'Multiply'}),
                **shaped_const_with_data('beta', None),
                **regular_op_with_empty_data('add', {'op': 'Add', 'type': 'Add'}),
                **result('result')
            },
            edges=[
                *connect('input', '0:squeeze'),
                *connect('squeeze_const', '1:squeeze'),
                *connect('squeeze', '0:mvn'),
                *connect('mvn_const', '1:mvn'),
                *connect('mvn', '0:unsqueeze'),
                *connect('unsqueeze_const', '1:unsqueeze'),
                *connect('unsqueeze', '0:mul'),
                *connect('gamma', '1:mul'),
                *connect('mul', '0:add'),
                *connect('beta', '1:add'),
                *connect('add', 'result')
            ]
        )
        LayerNormalization().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
