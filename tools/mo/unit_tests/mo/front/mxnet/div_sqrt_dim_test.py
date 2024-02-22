# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.mxnet.div_sqrt_dim import DivSqrtDim
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, shaped_parameter, regular_op_with_empty_data, result, connect, \
    shaped_const_with_data, connect_data, connect_front


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
                **shaped_const_with_data('gather_axis', None),
                **shaped_const_with_data('gather_indices', None),
                **regular_op_with_empty_data('gather', {'op': 'Gather', 'type': 'Gather'}),
                **regular_op_with_empty_data('power', {'op': 'AttributedPower', 'power': 0.5, 'type': 'Power'}),
                **regular_op_with_empty_data('cast', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.float32}),
                **regular_op_with_empty_data('div', {'op': 'Div', 'type': 'Divide'}),
                **result('result')
            },
            edges=[
                *connect('input', '0:div'),
                *connect_data('input', 'div_sqrt_shape_of'),
                *connect('div_sqrt_shape_of', '0:gather'),
                *connect('gather_axis', '1:gather'),
                *connect('gather_indices', '2:gather'),
                *connect('gather', 'cast'),
                *connect('cast', 'power'),
                *connect('power', '1:div'),
                *connect('div', 'result')
            ],
        )
        DivSqrtDim().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'result', 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
