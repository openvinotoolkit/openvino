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
                    'node_2': {'type': 'Identity', 'kind': 'op'},
                    'ctc': {'type': 'CTCGreedyDecoder', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'Result'},
                    }


class TestConcatPartialInfer(unittest.TestCase):
    def test_tf_ctc_greedy_decoder_nhwc_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'ctc'),
                                ('node_2', 'ctc'),
                                ('ctc', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([88, 2, 71])},
                                'node_2': {'shape': np.array([88, 2])},
                                'ctc': {'ctc_merge_repeated': 1}
                            })

        graph.graph['layout'] = 'NHWC'
        ctc_node = Node(graph, 'ctc')
        CTCGreedyDecoderOp.ctc_greedy_decoder_infer(ctc_node)
        exp_shape = np.array([2, 88, 1, 1])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_ctc_greedy_decoder_nchw_infer(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('node_1', 'ctc'),
                                ('node_2', 'ctc'),
                                ('ctc', 'node_3'),
                                ('node_3', 'op_output')
                            ],
                            {
                                'node_3': {'shape': None},
                                'node_1': {'shape': np.array([88, 2, 71])},
                                'node_2': {'shape': np.array([88, 2])},
                                'ctc': {'ctc_merge_repeated': 1}
                            })

        graph.graph['layout'] = 'NCHW'
        ctc_node = Node(graph, 'ctc')
        CTCGreedyDecoderOp.ctc_greedy_decoder_infer(ctc_node)
        exp_shape = np.array([2, 88, 1, 1])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
