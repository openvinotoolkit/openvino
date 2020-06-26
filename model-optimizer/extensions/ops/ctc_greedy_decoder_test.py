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

import numpy as np

from extensions.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_1_data': {'kind': 'data', 'value': None, 'shape': None},
                    'node_2': {'type': 'Identity', 'kind': 'op'},
                    'node_2_data': {'kind': 'data', 'value': None, 'shape': None},
                    'ctc': {'type': 'CTCGreedyDecoder', 'kind': 'op'},
                    'ctc_data': {'kind': 'data', 'value': None, 'shape': None},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'node_3_data': {'kind': 'data', 'value': None, 'shape': None},
                    'op_output': { 'kind': 'op', 'op': 'Result'},
                    }


class TestConcatPartialInfer(unittest.TestCase):
    def test_tf_ctc_greedy_decoder_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'node_1_data'),
                                ('node_1_data', 'ctc'),
                                ('node_2', 'node_2_data'),
                                ('node_2_data', 'ctc'),
                                ('ctc', 'ctc_data'),
                                ('ctc_data', 'node_3'),
                                ('node_3', 'node_3_data'),
                                ('node_3_data', 'op_output')
                            ],
                            {
                                'node_1_data': {'shape': np.array([88, 2, 71])},
                                'node_2_data': {'shape': np.array([88, 2])},
                                'ctc': {'ctc_merge_repeated': 1}
                            })

        graph.stage = 'middle'
        ctc_node = Node(graph, 'ctc')
        CTCGreedyDecoderOp.ctc_greedy_decoder_infer(ctc_node)
        exp_shape = np.array([2, 88, 1, 1])
        res_shape = graph.node['ctc_data']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
