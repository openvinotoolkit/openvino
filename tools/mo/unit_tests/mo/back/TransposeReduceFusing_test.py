# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.back.TransposeReduceFusing import TransposeReduce
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    # op
    'placeholder': {'kind': 'op', 'op': 'Parameter'},
    'transpose': {'kind': 'op', 'type': 'Transpose'},
    'reduceMean': {'kind': 'op', 'type': 'ReduceMean', 'keep_dims': False},
    'transpose_const': {'kind': 'op', 'type': 'Const'},
    'reduceMeanConst': {'kind': 'op', 'type': 'Const'},
    'convolution': {'kind': 'op', 'op': 'Convolution'},
    'gather': {'kind': 'op', 'op': 'Gather'},
    'gather_const': {'kind': 'op', 'op': 'Const'},

    # data
    'placeholder_data': {'kind': 'data'},
    'transpose_data': {'kind': 'data'},
    'reduceMean_data': {'kind': 'data'},
    'transpose_const_data': {'kind': 'data'},
    'reduceMeanConst_data': {'kind': 'data'},
    'gather_data': {'kind': 'data'},
    'gather_const_data': {'kind': 'data'}
}


class TestTransposeReduceFusing(unittest.TestCase):

    def test_positive(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('placeholder', 'placeholder_data'),
                                ('placeholder_data', 'transpose', {'in': 0}),
                                ('transpose_const', 'transpose_const_data'),
                                ('transpose_const_data', 'transpose', {'in': 1}),
                                ('transpose', 'transpose_data'),
                                ('transpose_data', 'reduceMean', {'in': 0}),
                                ('reduceMeanConst', 'reduceMeanConst_data'),
                                ('reduceMeanConst_data', 'reduceMean', {'in': 1}),
                                ('reduceMean', 'reduceMean_data'),
                                ('reduceMean_data', 'convolution')
                            ],
                            {
                                'transpose_const': {'value': int64_array([0, 2, 3, 1])},
                                'transpose_const_data': {'value': int64_array([0, 2, 3, 1])},
                                'reduceMeanConst': {'value': int64_array([1, 2])},
                                'reduceMeanConst_data': {'value': int64_array([1, 2])}
                            },
                            nodes_with_edges_only=True)
        ref_graph = build_graph(nodes_attributes,
                                [
                                    ('placeholder', 'placeholder_data'),
                                    ('placeholder_data', 'reduceMean', {'in': 0}),
                                    ('transpose_const', 'transpose_const_data'),
                                    ('transpose_const_data', 'gather', {'in': 0}),
                                    ('reduceMeanConst', 'reduceMeanConst_data'),
                                    ('reduceMeanConst_data', 'gather', {'in': 1}),
                                    ('gather_const', 'gather_const_data'),
                                    ('gather_const_data', 'gather', {'in': 2}),
                                    ('gather', 'gather_data'),
                                    ('gather_data', 'reduceMean', {'in': 1}),
                                    ('reduceMean', 'reduceMean_data'),
                                    ('reduceMean_data', 'convolution')
                                ],
                                {
                                    'transpose_const_data': {'value': int64_array([0, 2, 3, 1])},
                                    'reduceMeanConst_data': {'value': int64_array([1, 2])},
                                },
                                nodes_with_edges_only=True)
        TransposeReduce().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative_values(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('placeholder', 'placeholder_data'),
                                ('placeholder_data', 'transpose', {'in': 0}),
                                ('transpose_const', 'transpose_const_data'),
                                ('transpose_const_data', 'transpose', {'in': 1}),
                                ('transpose', 'transpose_data'),
                                ('transpose_data', 'reduceMean', {'in': 0}),
                                ('reduceMeanConst', 'reduceMeanConst_data'),
                                ('reduceMeanConst_data', 'reduceMean', {'in': 1}),
                                ('reduceMean', 'reduceMean_data'),
                                ('reduceMean_data', 'convolution')
                            ],
                            {
                                'transpose_const': {'value': int64_array([0, 1, 3, 2])},
                                'transpose_const_data': {'value': int64_array([0, 1, 3, 2])},
                                'reduceMeanConst': {'value': int64_array([1])},
                                'reduceMeanConst_data': {'value': int64_array([1])}
                            },
                            nodes_with_edges_only=True)
        ref_graph = build_graph(nodes_attributes,
                                [
                                    ('placeholder', 'placeholder_data'),
                                    ('placeholder_data', 'transpose', {'in': 0}),
                                    ('transpose_const', 'transpose_const_data'),
                                    ('transpose_const_data', 'transpose', {'in': 1}),
                                    ('transpose', 'transpose_data'),
                                    ('transpose_data', 'reduceMean', {'in': 0}),
                                    ('reduceMeanConst', 'reduceMeanConst_data'),
                                    ('reduceMeanConst_data', 'reduceMean', {'in': 1}),
                                    ('reduceMean', 'reduceMean_data'),
                                    ('reduceMean_data', 'convolution')
                                ],
                                {
                                    'transpose_const': {'value': int64_array([0, 1, 3, 2])},
                                    'transpose_const_data': {'value': int64_array([0, 1, 3, 2])},
                                    'reduceMeanConst': {'value': int64_array([1])},
                                    'reduceMeanConst_data': {'value': int64_array([1])}
                                },
                                nodes_with_edges_only=True)
        TransposeReduce().find_and_replace_pattern(graph)
        flag, resp = compare_graphs(graph, ref_graph, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('placeholder', 'placeholder_data'),
                                ('placeholder_data', 'reduceMean', {'in': 0}),
                                ('reduceMeanConst', 'reduceMeanConst_data'),
                                ('reduceMeanConst_data', 'reduceMean', {'in': 1}),
                                ('reduceMean', 'reduceMean_data'),
                                ('reduceMean_data', 'convolution')
                            ],
                            nodes_with_edges_only=True)
        ref_graph = build_graph(nodes_attributes,
                                [
                                    ('placeholder', 'placeholder_data'),
                                    ('placeholder_data', 'reduceMean', {'in': 0}),
                                    ('reduceMeanConst', 'reduceMeanConst_data'),
                                    ('reduceMeanConst_data', 'reduceMean', {'in': 1}),
                                    ('reduceMean', 'reduceMean_data'),
                                    ('reduceMean_data', 'convolution')
                                ],
                                nodes_with_edges_only=True)

        TransposeReduce().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'convolution', check_op_attrs=True)
        self.assertTrue(flag, resp)
