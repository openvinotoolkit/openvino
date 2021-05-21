# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from extensions.middle.TFDepthwiseConv2dNativeReshape import TFDepthwiseConv2dNativeReshape
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attrs = {
    'in_data': {'kind': 'data'},
    'conv': {'kind': 'op', 'op': 'DepthwiseConv2dNative', 'input_feature_channel': 2},
    'kernel': {'kind': 'op', 'op': 'Const'},
    'kernel_data': {'kind': 'data', 'shape': int64_array([7, 7, 1, 8])},

    'reshape': {'kind': 'op', 'op': "Reshape"},
    'reshape_data': {'kind': 'data'},
    'reshape_const': {'kind': 'op', 'op': 'Const'},
    'reshape_const_data': {'kind': 'data', 'shape': int64_array([4]), 'value': int64_array([0, 0, -1, 1])}
}


class TFDepthwiseConv2dNativeReshapeTest(unittest.TestCase):

    def test_1(self):
        graph = build_graph(nodes_attrs=nodes_attrs,
                            edges=[
                                ('kernel', 'kernel_data'),
                                ('in_data', 'conv', {'in': 0}),
                                ('kernel_data', 'conv', {'in': 1})
                            ],
                            nodes_with_edges_only=True)

        TFDepthwiseConv2dNativeReshape().find_and_replace_pattern(graph)
        ref_graph = build_graph(nodes_attrs=nodes_attrs,
                                edges=[
                                    ('kernel', 'kernel_data'),
                                    ('in_data', 'conv', {'in': 0}),
                                    ('kernel_data', 'reshape', {'in': 0}),
                                    ('reshape_const', 'reshape_const_data'),
                                    ('reshape_const_data', 'reshape', {'in': 1}),
                                    ('reshape', 'reshape_data'),
                                    ('reshape_data', 'conv', {'in': 1})
                                ])
        (flag, resp) = compare_graphs(graph, ref_graph, 'conv', check_op_attrs=True)
        self.assertTrue(flag, resp)
