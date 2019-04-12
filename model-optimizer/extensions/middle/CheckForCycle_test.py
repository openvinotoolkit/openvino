"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.middle.CheckForCycle import CheckForCycle
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_3_data': {'value': None, 'kind': 'data', 'data_type': None},
                    # Placeholders
                    'placeholder_1': {'shape': None, 'type': 'Input', 'kind': 'op', 'op': 'Placeholder'},
                    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2': {'shape': None, 'type': 'Input', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_1': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'pl_2': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_2_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    # ScaleShift layer
                    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
                    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    # Mul op
                    'mul_1': {'type': None, 'kind': 'op', 'op': 'Mul'},
                    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'OpOutput', 'infer': lambda x: None}
                    }


class CycleTest(unittest.TestCase):
    def test_check_for_cycle1(self):
        # cyclic case
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_1_data'),
                             ('node_1_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('node_3_data', 'node_1')],
                            nodes_with_edges_only=True)
        with self.assertRaisesRegex(Error, 'Graph contains a cycle. Can not proceed.*'):
            CheckForCycle().find_and_replace_pattern(graph)

    def test_check_for_cycle2(self):
        # acyclic case
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_1_data'),
                             ('node_1_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('node_3_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data')
                             ],
                            nodes_with_edges_only=True)
        try:
            CheckForCycle().find_and_replace_pattern(graph)
        except Error:
            self.fail("Unexpected Error raised")
