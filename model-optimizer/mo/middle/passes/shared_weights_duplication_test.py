"""
 Copyright (c) 2018 Intel Corporation

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

from mo.middle.passes.shared_weights_duplication import duplicate_shared_weights
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    # Mul and Add operations
    'mul_1': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_2': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'mul_2_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_3': {'type': None, 'kind': 'op', 'op': 'Mul'},
    'mul_3_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Concat1 operation
    'concat_1': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
    'concat_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class DuplicateSharedWeightsTests(unittest.TestCase):
    def test_duplicate_shared_weights_1(self):
        graph = build_graph(nodes_attributes,
                            [('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data'),
                             ('mul_1_w', 'mul_2'),
                             ('mul_2', 'mul_2_data'),
                             ('mul_1_w', 'mul_3'),
                             ('mul_3', 'mul_3_data'),
                             ('mul_1_data', 'concat_1'),
                             ('mul_2_data', 'concat_1'),
                             ('mul_3_data', 'concat_1'),
                             ('concat_1', 'concat_1_data')
                             ],
                            {'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])}})

        graph_ref = build_graph(nodes_attributes,
                                [('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_2_w', 'mul_2'),
                                 ('mul_2', 'mul_2_data'),
                                 ('mul_3_w', 'mul_3'),
                                 ('mul_3', 'mul_3_data'),
                                 ('mul_1_data', 'concat_1'),
                                 ('mul_2_data', 'concat_1'),
                                 ('mul_3_data', 'concat_1'),
                                 ('concat_1', 'concat_1_data')
                                 ],
                                {'mul_1_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_2_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 'mul_3_w': {'shape': np.array([3]), 'value': np.array([1, 2, 3])},
                                 })

        duplicate_shared_weights(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat_1_data')
        self.assertTrue(flag, resp)
