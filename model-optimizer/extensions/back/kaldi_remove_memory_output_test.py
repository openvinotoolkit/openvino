# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
