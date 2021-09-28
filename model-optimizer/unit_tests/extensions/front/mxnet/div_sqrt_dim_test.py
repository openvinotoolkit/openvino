# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.front.mxnet.div_sqrt_dim import DivSqrtDim
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, shaped_parameter, result, connect, regular_op_with_empty_data, \
    connect_data, shaped_const_with_data


class DivSqrtDimTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **regular_op_with_empty_data('div_sqrt_dim', {'op': '_contrib_div_sqrt_dim'}),
                **result('result')
            },
            edges=[
                *connect('input', 'div_sqrt_dim'),
                *connect('div_sqrt_dim', 'result')
            ]
        )

        ref_graph = build_graph(
            nodes_attrs={
                **shaped_parameter('input', int64_array([1, 3, 15, 15])),
                **regular_op_with_empty_data('div_sqrt_shape_of', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
                **regular_op_with_empty_data('div_sqrt_rank', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
                **shaped_const_with_data('const_1', None),
                **regular_op_with_empty_data('add_1', {'op': 'Add', 'type': 'Add'}),
                **shaped_const_with_data('const_2', None),
                **regular_op_with_empty_data('add_2', {'op': 'Add', 'type': 'Add'}),
                **shaped_const_with_data('const_3', None),
                **regular_op_with_empty_data('add_3', {'op': 'Add', 'type': 'Add'}),
                **shaped_const_with_data('const_4', None),
                **regular_op_with_empty_data('add_4', {'op': 'Add', 'type': 'Add'}),
                **shaped_const_with_data('const_5', None),
                **regular_op_with_empty_data('range', {'op': 'Range', 'type': 'Range'}),
                **shaped_const_with_data('gather_axis', None),
                **regular_op_with_empty_data('gather', {'op': 'Gather', 'type': 'Gather'}),
                **regular_op_with_empty_data('power', {'op': 'AttributedPower', 'power': 0.5, 'type': 'Power'}),
                **regular_op_with_empty_data('div', {'op': 'Div', 'type': 'Divide'}),
                **result('result')
            },
            edges=[
                *connect('input', 'div_sqrt_shape_of'),
                *connect('div_sqrt_shape_of', 'div_sqrt_rank'),
                *connect('div_sqrt_rank', '0:add_1'),
                *connect('const_1', '1:add_1'),
                *connect_data('div_sqrt_rank', '0:add_2'),
                *connect('const_2', '1:add_2'),
                *connect('add_1', '0:add_3'),
                *connect('const_3', '1:add_3'),
                *connect('add_2', '0:add_4'),
                *connect('const_4', '1:add_4'),
                *connect('add_3', '1:range'),
                *connect('const_5', '2:range'),
                *connect('add_4', '0:range'),
                *connect_data('div_sqrt_shape_of', '0:gather'),
                *connect('range', '1:gather'),
                *connect('gather_axis', '2:gather'),
                *connect('gather', 'power'),
                *connect('power', '1:div'),
                *connect_data('input', '0:div'),
                *connect('div', 'result')
            ],
        )
        DivSqrtDim().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
