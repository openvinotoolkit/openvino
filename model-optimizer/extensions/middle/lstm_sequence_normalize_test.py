
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

from extensions.middle.lstm_sequence_normalize import LSTMSequenceNormalize
from mo.utils.unittest.graph import compare_graphs, build_graph_with_attrs
from mo.graph.graph import Node


class LSTMSequenceNormalizeTest(unittest.TestCase):

    def test_squeeze_num_directions(self):
        tested_obj = LSTMSequenceNormalize()
        pattern = tested_obj.pattern()
        orig_shape = np.array([10, 1, 20, 128], dtype=np.int64)  # seq_length, num_dims, batch_size, data_size
        new_shape = np.array([10, 20, 128], dtype=np.int64)
        graph = build_graph_with_attrs(
            nodes_with_attrs=pattern['nodes'],
            edges_with_attrs=pattern['edges'],
            update_edge_attrs={
                ('W', 'lstm', 0): {'in': 1},
                ('R', 'lstm', 0): {'in': 2},
            },
            new_nodes_with_attrs=[
                ('output', {'shape': orig_shape}),
            ],
            new_edges_with_attrs=[
                ('lstm', 'output', {'out': 0}),
            ],
        )

        lstm = Node(graph, 'lstm')
        match = {'lstm': lstm}
        tested_obj.squeeze_num_directions(graph, match)
        self.assertTrue(np.array_equal(lstm.out_node(0).shape, new_shape))
        reshape_node = lstm.out_node(0).out_node(0)
        self.assertTrue(reshape_node.op == 'Reshape')
        self.assertTrue(np.array_equal(reshape_node.dim, orig_shape))
        self.assertTrue(reshape_node.out_node(0).id == 'output')
