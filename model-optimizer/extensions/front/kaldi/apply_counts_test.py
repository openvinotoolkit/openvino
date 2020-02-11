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
from extensions.front.kaldi.apply_counts import apply_biases_to_last_layer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


class TestKaldiPipeline(unittest.TestCase):
    def test_apply_biases_to_ScaleShift(self):
        counts = -0.5 * np.ones(10)
        nodes = {'input': {'kind': 'op', 'op': None},
                 'weights': {'kind': 'op', 'op': 'Const'},
                 'biases': {'kind': 'op', 'op': 'Const', 'value': None, 'shape': None, 'data_type': None},
                 'sc': {'op': 'ScaleShift', 'kind': 'op'},
                 'sub': {'op': 'Add', 'kind': 'op'},
                 "const": {'op': 'Const', 'value': -counts, 'kind': 'op'},
                 'op_output': {'op': 'Result', 'kind': 'op'}
                 }
        graph = build_graph(nodes,
                            [
                                ('input', 'sc', {'in': 0}),
                                ('weights', 'sc', {'in': 1}),
                                ('biases', 'sc', {'in': 2}),
                                ('sc', 'op_output')
                            ], nodes_with_edges_only=True)

        graph.stage = "front"
        ref_graph = build_graph(nodes,
                            [
                                ('input', 'sc', {'in': 0}),
                                ('weights', 'sc', {'in': 1}),
                                ('biases', 'sc', {'in': 2}),
                                ('sc', 'sub', {'in': 0}),
                                ('const', 'sub', {'in': 1}),
                                ('sub', 'op_output')
                            ], nodes_with_edges_only=True)

        apply_biases_to_last_layer(graph, counts)
        compare_graphs(graph, ref_graph, 'op_output', check_op_attrs=True)

    def test_apply_biases_to_graph_with_SoftMax(self):
        counts = -0.5 * np.ones(10)
        nodes = {'input': {'kind': 'op', 'op': None},
                 'weights': {'kind': 'op', 'op': 'Const'},
                 'biases': {'kind': 'op', 'op': 'Const', 'value': None, 'shape': None, 'data_type': None},
                 'fc': {'op': 'FullyConnected', 'kind': 'op'},
                 'softmax': {'op': 'SoftMax', 'kind': 'op'},
                 'op_output': {'op': 'Result', 'kind': 'op'},
                 'sub': {'op': 'Add', 'kind': 'op'},
                 "const": {'op': 'Const', 'value': -counts, 'kind': 'op'},
                 }
        graph = build_graph(nodes,
                            [
                                ('input', 'fc', {'in': 0}),
                                ('weights', 'fc', {'in': 1}),
                                ('biases', 'fc', {'in': 2}),
                                ('fc', 'softmax'),
                                ('softmax','op_output')
                            ], nodes_with_edges_only=True)
        ref_graph = build_graph(nodes,
                            [
                                ('input', 'fc', {'in': 0}),
                                ('weights', 'fc', {'in': 1}),
                                ('biases', 'fc', {'in': 2}),
                                ('fc', 'sub', {'in': 0}),
                                ('const', 'sub', {'in': 1}),
                                ('sub', 'op_output')
                            ], nodes_with_edges_only=True)

        graph.stage = "front"
        apply_biases_to_last_layer(graph, counts)
        compare_graphs(graph, ref_graph, 'op_output', check_op_attrs=True)
