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

from extensions.middle.ReplacePNorm import ReplacePNormNodePattern
from mo.utils.unittest.graph import build_graph, compare_graphs


class ReplacePNormNodePatternTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes_attributes = {
            'placeholder': {'kind': 'op', 'op': None},
            'in_node': {'kind': 'data', 'shape': [1, 3500]},
            'pnorm': {'kind': 'op', 'op': 'pnorm', 'group': 10, 'p': 2.0},
            'pnorm_data': {'kind': 'data', 'shape': [1, 350]},
            'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
        }

    def test_pnorm(self):
        graph = build_graph(self.nodes_attributes,
                            [('placeholder', 'in_node'),
                             ('in_node', 'pnorm'),
                             ('pnorm', 'pnorm_data'),
                             ('pnorm_data', 'out_placeholder')])
        ReplacePNormNodePattern().find_and_replace_pattern(graph)

        ref_graph = build_graph({'in_placeholder': {'kind': 'op', 'op': None},
                                 'in_node': {'kind': 'data', 'shape': [1, 3500]},
                                 'power': {'kind': 'op', 'op': 'Power', 'power': 2.0},
                                 'power_data': {'kind': 'data'},
                                 'reshape':  {'kind': 'op', 'op': 'Reshape'},
                                 'reshape_data': {'kind': 'data'},
                                 'const': {'kind': 'op', 'op': 'Const', 'value': [1, 350, 10]},
                                 'const_data': {'kind': 'data'},
                                 'reduce': {'kind': 'op', 'op': 'ReduceSum'},
                                 'reduce_data': {'kind': 'data'},
                                 'const_1': {'kind': 'op', 'op': 'Const', 'value': 2},
                                 'const_data_1': {'kind': 'data'},
                                 'invpower': {'kind': 'op', 'op': 'Power', 'power': 0.5},
                                 'invpower_data': {'kind': 'data'},
                                 'out_placeholder': {'kind': 'op', 'op': 'placeholder'},
                                 },
                                [
                                    ('in_placeholder', 'in_node'),
                                    ('in_node', 'power'),
                                    ('power', 'power_data'),
                                    ('power_data', 'reshape', {'in': 0}),
                                    ('reshape', 'reshape_data'),
                                    ('const', 'const_data'),
                                    ('const_data', 'reshape', {'in': 1}),
                                    ('reshape_data', 'reduce', {'in': 0}),
                                    ('const_1', 'const_data_1'),
                                    ('const_data_1', 'reduce', {'in': 1}),
                                    ('reduce', 'reduce_data'),
                                    ('reduce_data', 'invpower'),
                                    ('invpower', 'invpower_data'),
                                    ('invpower_data', 'out_placeholder'),
                                ]
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_placeholder')
        self.assertTrue(flag, resp)
