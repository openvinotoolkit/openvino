# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.back.insert_compatibility_l2normalization import CompatibilityL2NormalizationPattern
from unit_tests.utils.graph import build_graph


class CompatibilityL2NormalizationPatternTest(unittest.TestCase):
    nodes = {
        'input_node': {
            'kind': 'data'
        },
        'l2norm_node': {
            'op': 'Normalize',
            'kind': 'op',
            'type': 'Normalize',
        },
        'output_node': {
            'kind': 'data'
        }
    }

    def test_insert_data(self):
        graph = build_graph(self.nodes, [('input_node', 'l2norm_node'), ('l2norm_node', 'output_node')],
                            {'input_node': {'shape': np.array([1, 10])},
                             })
        CompatibilityL2NormalizationPattern().find_and_replace_pattern(graph)
        self.assertEqual(len(graph.nodes()), 5)
        self.assertEqual(graph.node['l2norm_node_weights']['name'], 'l2norm_node_weights')
        self.assertEqual(len(graph.node['l2norm_node_weights']['value']), 10)

        expect_value = np.full([10], 1.0, np.float32)

        for i, val in enumerate(expect_value):
            self.assertEqual(graph.node['l2norm_node_weights']['value'][i], val)
