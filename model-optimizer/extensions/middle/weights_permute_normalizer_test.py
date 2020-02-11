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

from extensions.middle.UselessStridedSlice import UselessStridedSliceEraser
from extensions.middle.wights_permute_normalizer import WeightsPermuteNormalizer
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

nodes_attributes = {
    'placeholder': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_data': {'value': None, 'shape': None, 'kind': 'data'},

    'quantize': {'type': 'FakeQuantize', 'kind': 'op', 'op': 'FakeQuantize'},
    'quantize_data': {'value': None, 'shape': None, 'kind': 'data'},

    'const_1': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
    'const_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'const_2': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
    'const_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    'const_3': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
    'const_3_data': {'value': None, 'shape': None, 'kind': 'data'},

    'const_4': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
    'const_4_data': {'value': None, 'shape': None, 'kind': 'data'},

    'const_5': {'type': 'Const', 'kind': 'op', 'op': 'Const'},
    'const_5_data': {'value': None, 'shape': None, 'kind': 'data'},

    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2D', 'layout': 'NHWC'},
    'conv_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'conv_1_b': {'value': None, 'shape': None, 'kind': 'data'},
    'const_conv_1_w': {'value': None, 'shape': None, 'kind': 'op'},
    'const_conv_1_b': {'value': None, 'shape': None, 'kind': 'op'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class WeightNormalizationTests(unittest.TestCase):
    def test_normalize_weights_test1(self):
        #   FakeQuantize---,->Conv
        #  Placeholder--'
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('const_1', 'const_1_data'),
                             ('const_2', 'const_2_data'),
                             ('const_3', 'const_3_data'),
                             ('const_4', 'const_4_data'),
                             ('const_5', 'const_5_data'),
                             ('quantize', 'quantize_data'),
                             ('conv_1', 'conv_1_data'),
                             ('const_1_data', 'quantize'),
                             ('const_2_data', 'quantize'),
                             ('const_3_data', 'quantize'),
                             ('const_4_data', 'quantize'),
                             ('const_5_data', 'quantize'),
                             ('placeholder_data', 'conv_1'),
                             ('quantize_data', 'conv_1', {'in': 1, 'permutation': "[3, 2, 0, 1]"}),
                             ],
                            {},
                            nodes_with_edges_only=True
                            )

        pattern = WeightsPermuteNormalizer()
        pattern.find_and_replace_pattern(graph)

        conv = Node(graph, 'conv_1')
        quantize = Node(graph, 'quantize')

        self.assertTrue('permutation' in conv.in_edge(1) and conv.in_edge(1)['permutation'] == "[3, 2, 0, 1]")
        self.assertTrue('permutation' in quantize.in_edge(0) and quantize.in_edge(0)['permutation'] == "[3, 2, 0, 1]")

    def test_normalize_weights_test2(self):
        #     Quantize---,->Conv
        #  Placeholder--'
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'placeholder_data'),
                             ('const_1', 'const_1_data'),
                             ('const_2', 'const_2_data'),
                             ('const_3', 'const_3_data'),
                             ('const_4', 'const_4_data'),
                             ('const_5', 'const_5_data'),
                             ('quantize', 'quantize_data'),
                             ('conv_1', 'conv_1_data'),
                             ('const_1_data', 'quantize'),
                             ('const_2_data', 'quantize'),
                             ('const_3_data', 'quantize'),
                             ('const_4_data', 'quantize'),
                             ('const_5_data', 'quantize'),
                             ('quantize_data', 'conv_1', {'in': 0}),
                             ('conv_1_w', 'conv_1', {'in': 1}),
                             ],
                            {},
                            nodes_with_edges_only=True
                            )

        pattern = WeightsPermuteNormalizer()
        pattern.find_and_replace_pattern(graph)

        conv = Node(graph, 'conv_1')
        quantize = Node(graph, 'quantize')

        self.assertTrue('permutation' not in conv.in_edge(1))
        self.assertTrue('permutation' not in quantize.in_edge(0))
