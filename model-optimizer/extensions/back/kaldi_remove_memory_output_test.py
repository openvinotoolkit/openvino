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

from extensions.back.kaldi_remove_memory_output import KaldiRemoveMemoryOutputBackReplacementPattern
from mo.utils.unittest.graph import build_graph


class KaldiRemoveMemoryOutputTest(unittest.TestCase):
    nodes = {
        'input_node': {
            'kind': 'data'
        },
        'memory_node': {
            'op': 'Assign',
            'kind': 'op'
        },
        'output_node': {
            'kind': 'data'
        },
        'op_output': {
            'kind': 'data',
            'op': 'Result',
        }
    }

    def test_remove_out_data_for_memory(self):
        graph = build_graph(self.nodes,
                            [
                                ('input_node', 'memory_node'),
                                ('memory_node', 'output_node'),
                                ('output_node', 'op_output')
                            ])
        KaldiRemoveMemoryOutputBackReplacementPattern().find_and_replace_pattern(graph)
        self.assertNotIn('output_node', graph.node)

    def test_do_not_remove_out_data_for_memory(self):
        graph = build_graph(self.nodes,
                            [
                                ('input_node', 'memory_node'),
                                ('memory_node', 'output_node'),
                            ])
        KaldiRemoveMemoryOutputBackReplacementPattern().find_and_replace_pattern(graph)
        self.assertIn('output_node', graph.node)
