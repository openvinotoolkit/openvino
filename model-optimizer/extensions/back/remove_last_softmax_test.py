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

from extensions.back.remove_last_softmax_pattern import RemoveLastSoftMaxPattern, RemoveLastLogSoftMaxPattern
from mo.utils.unittest.graph import build_graph


class KaldiRemoveLastSoftMaxTest(unittest.TestCase):
    nodes = {
        'input_node': {
            'kind': 'data'
        },
        'softmax_node': {
            'op': 'SoftMax',
            'kind': 'op'
        },
        'output_node': {
            'kind': 'data'
        },
        'op_output': {
            'kind': 'op',
            'op': 'Result'
        },
        'log_node': {
            'op': 'Log',
            'kind': 'op'
        },
        'log_data': {
            'kind': 'data'
        },
    }

    def test_remove_last_SoftMax(self):
        graph = build_graph(self.nodes, [
            ('input_node', 'softmax_node'),
            ('softmax_node', 'output_node'),
            ('output_node', 'op_output')
        ], nodes_with_edges_only=True)
        RemoveLastSoftMaxPattern().find_and_replace_pattern(graph)
        self.assertNotIn('softmax_node', graph.node)

    def test_remove_last_LogSoftMax(self):
        graph = build_graph(self.nodes, [
            ('input_node', 'softmax_node'),
            ('softmax_node', 'output_node'),
            ('output_node', 'log_node'),
            ('log_node', 'log_data'),
            ('log_data', 'op_output')
        ], nodes_with_edges_only=True)
        RemoveLastLogSoftMaxPattern().find_and_replace_pattern(graph)
        self.assertNotIn('softmax_node', graph.node)
        self.assertNotIn('log_node', graph.node)

    def test_do_not_remove_not_last_SoftMax(self):
        graph = build_graph(self.nodes, [
            ('input_node', 'softmax_node'),
            ('softmax_node', 'output_node')
        ])
        RemoveLastSoftMaxPattern().find_and_replace_pattern(graph)
        self.assertIn('softmax_node', graph.node)
