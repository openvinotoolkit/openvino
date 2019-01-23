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

from extensions.front.reciprocal import ReciprocalReplacer
from mo.utils.unittest.graph import build_graph, compare_graphs


class ReciprocalReplacerTests(unittest.TestCase):
    @staticmethod
    def _create_graphs():
        return (
            build_graph(
                {'placeholder': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
                 'reciprocal': {'kind': 'op', 'op': 'Reciprocal'}},
                [('placeholder', 'reciprocal')]),

            build_graph(
                {'placeholder': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
                 'power': {'type': 'Power', 'kind': 'op', 'op': 'Power', 'scale': 1, 'power': -1, 'shift': 0}},
                [('placeholder', 'power')])
        )

    def test_replace_reciprocal(self):
        graph, graph_ref = __class__._create_graphs()

        pattern = ReciprocalReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'reciprocal/power_', last_node_ref='power', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_neg_replace_reciprocal(self):
        graph, graph_ref = __class__._create_graphs()
        graph_ref.node['power']['power'] = 0

        pattern = ReciprocalReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'reciprocal/power_', last_node_ref='power', check_op_attrs=True)
        self.assertTrue(not flag)