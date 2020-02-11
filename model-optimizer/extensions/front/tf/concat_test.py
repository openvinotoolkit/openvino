"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.front.tf.concat import Concat
from mo.utils.unittest.graph import build_graph_with_edge_attrs


class TestConcatEdgesReshuffler(unittest.TestCase):
    def test_concat_edges_reshaffle(self):
        graph = build_graph_with_edge_attrs(
            {'axis': {},
             'input_1': {},
             'input_2': {},
             'input_3': {},
             'concat': {'op': 'Concat', 'simple_concat': True, 'axis': 1},
             },
            [('axis', 'concat', {'in': 0}),
             ('input_1', 'concat', {'in': 1}),
             ('input_2', 'concat', {'in': 2}),
             ('input_3', 'concat', {'in': 3})],
        )
        Concat().find_and_replace_pattern(graph=graph)
        for u, v, attrs in graph.in_edges('concat', data=True):
            if attrs['in'] == 0:
                self.assertEqual(u, 'input_1')
            if attrs['in'] == 1:
                self.assertEqual(u, 'input_2')
            if attrs['in'] == 2:
                self.assertEqual(u, 'input_3')
            if attrs['in'] == 3:
                self.assertEqual(u, 'axis')
        self.assertTrue('axis' not in graph.node['concat'])
