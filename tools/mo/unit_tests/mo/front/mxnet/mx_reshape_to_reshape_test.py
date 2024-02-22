# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.mx_reshape_to_reshape import MXReshapeToReshape
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from unit_tests.utils.graph import build_graph


class TestMXReshapeToReshape(unittest.TestCase):
    def test_minus2(self):
        graph = build_graph({'node_1': {'shape': int64_array([1, 2, 3, 4]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                             'reshape': {'kind': 'op', 'op': 'MXReshape', 'dim': int64_array([1, 2, -2]), 'reverse': False},
                             'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
                             },
                            [('node_1', 'reshape', {'in': 0}),
                             ('reshape', 'last', {'in': 0}), ],
                             nodes_with_edges_only=True
                            )

        graph.stage = 'front'
        pattern = MXReshapeToReshape()
        pattern.find_and_replace_pattern(graph)
        graph.clean_up()
        reshape_count = 0
        concat_count = 0
        mxreshape_count = 0

        nodes = graph.get_op_nodes()
        for node in nodes:
            if node['op'] == 'Reshape':
                reshape_count = reshape_count + 1
            elif node['op'] == 'MXReshape':
                mxreshape_count = mxreshape_count + 1
            elif node['op'] == 'Concat':
                concat_count = concat_count + 1

        self.assertTrue(reshape_count == 1)
        self.assertTrue(concat_count == 1)
        self.assertTrue(mxreshape_count == 0)


    def test_minus3(self):
        graph = build_graph({'node_1': {'shape': int64_array([1, 2, 3, 4]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                             'reshape': {'kind': 'op', 'op': 'MXReshape', 'dim': int64_array([1, -3, 4]), 'reverse': False},
                             'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
                             },
                            [('node_1', 'reshape', {'in': 0}),
                             ('reshape', 'last', {'in': 0}), ],
                             nodes_with_edges_only=True
                            )

        graph.stage = 'front'
        pattern = MXReshapeToReshape()
        pattern.find_and_replace_pattern(graph)
        graph.clean_up()
        reshape_count = 0
        concat_count = 0
        mxreshape_count = 0

        nodes = graph.get_op_nodes()
        for node in nodes:
            if node['op'] == 'Reshape':
                reshape_count = reshape_count + 1
            elif node['op'] == 'MXReshape':
                mxreshape_count = mxreshape_count + 1
            elif node['op'] == 'Concat':
                concat_count = concat_count + 1

        self.assertTrue(reshape_count == 1)
        self.assertTrue(concat_count == 1)
        self.assertTrue(mxreshape_count == 0)


    def test_minus4(self):
        graph = build_graph({'node_1': {'shape': int64_array([1, 6]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                             'reshape': {'kind': 'op', 'op': 'MXReshape', 'dim': int64_array([1, -4, 2, 3, 1]), 'reverse': False},
                             'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
                             },
                            [('node_1', 'reshape', {'in': 0}),
                             ('reshape', 'last', {'in': 0}), ],
                             nodes_with_edges_only=True
                            )

        graph.stage = 'front'
        pattern = MXReshapeToReshape()
        pattern.find_and_replace_pattern(graph)
        graph.clean_up()
        reshape_count = 0
        concat_count = 0
        mxreshape_count = 0

        nodes = graph.get_op_nodes()
        for node in nodes:
            if node['op'] == 'Reshape':
                reshape_count = reshape_count + 1
            elif node['op'] == 'MXReshape':
                mxreshape_count = mxreshape_count + 1
            elif node['op'] == 'Concat':
                concat_count = concat_count + 1

        self.assertTrue(reshape_count == 1)
        self.assertTrue(concat_count == 1)
        self.assertTrue(mxreshape_count == 0)
