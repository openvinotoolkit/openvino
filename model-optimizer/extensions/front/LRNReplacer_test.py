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

from extensions.front.LRNReplacer import LRNReplacer
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # LRN operation
    'LRN': {'type': None, 'kind': 'op', 'op': 'LRN'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Mul operation
    'mul_1': {'type': None, 'value': None, 'kind': 'op', 'op': 'Mul'},
    'const_mul_1_w': {'type': None, 'value': None, 'kind': 'op', 'op': 'Const'},
}


class LRNReplacerTest(unittest.TestCase):
    def test_LRNreplacer_test_1(self):
        # Test with Mul operation
        alpha = 0.0002
        beta = 0.5
        bias = 2.0
        nsize = 3

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'LRN'),
                             ('LRN', 'last'),
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'LRN': {'alpha': np.array(alpha), 'beta': np.array(beta),
                                     'bias': np.array(bias), 'size': np.array(nsize)},
                             }, nodes_with_edges_only=True)

        alpha /= bias
        scale_value = np.array(1. / (pow(bias, beta)))

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'LRN'),
                                 ('LRN', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1'),
                                 ('mul_1', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'LRN': {'alpha': np.array(alpha), 'beta': np.array(beta),
                                         'size': np.array(nsize)},
                                 'const_mul_1_w': {'shape': scale_value.shape,
                                                   'value': scale_value},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = LRNReplacer()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_LRNreplacer_test_2(self):
        # Test without Mul operation
        alpha = 0.0002
        beta = 0.5
        bias = 1
        nsize = 3

        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'LRN'),
                             ('LRN', 'last'),
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'LRN': {'alpha': np.array(alpha), 'beta': np.array(beta),
                                     'bias': np.array(bias), 'size': np.array(nsize)},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'LRN'),
                                 ('LRN', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'LRN': {'alpha': np.array(alpha), 'beta': np.array(beta),
                                         'bias': np.array(bias), 'size': np.array(nsize)},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = LRNReplacer()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
