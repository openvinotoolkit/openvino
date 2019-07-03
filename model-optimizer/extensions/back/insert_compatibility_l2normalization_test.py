"""
 Copyright (c) 2017-2019 Intel Corporation

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

import numpy as np

from extensions.back.insert_compatibility_l2normalization import CompatibilityL2NormalizationPattern
from mo.utils.unittest.graph import build_graph


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
        self.assertEqual(len(graph.nodes()), 4)
        self.assertEqual(graph.node['l2norm_node_weights']['name'], 'l2norm_node_weights')
        self.assertEqual(len(graph.node['l2norm_node_weights']['value']), 10)

        expect_value = np.full([10], 1.0, np.float32)

        for i, val in enumerate(expect_value):
            self.assertEqual(graph.node['l2norm_node_weights']['value'][i], val)
