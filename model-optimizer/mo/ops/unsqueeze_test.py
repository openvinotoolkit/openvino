"""
 Copyright (c) 2018-2019 Intel Corporation

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
from generator import generator

from mo.graph.graph import Node
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.unittest.graph import build_graph, compare_graphs


@generator
class TestUnsqueezeOp(unittest.TestCase):
    nodes_attributes = {
        'data_1': {
            'kind': 'data',
            'shape': None,
            'value': None,
        },
        'unsq': {
            'op': 'Unsqueeze',
            'kind': 'op',
            'unsqueeze_dims': None,
        },
        'data_2': {
            'kind': 'data',
            'shape': None,
            'value': None,
        }
    }

    def test_unsqueeze_infer(self):
        graph = build_graph(self.nodes_attributes,
                            [('data_1', 'unsq'),
                             ('unsq', 'data_2')],
                            {'data_1': {'shape': np.array([1, 3, 64, 64])},
                             'unsq': {'unsqueeze_dims': np.array([0, 4])}
                             })

        graph_ref = build_graph(self.nodes_attributes,
                                [('data_1', 'unsq'),
                                 ('unsq', 'data_2')],
                                {'data_1': {'shape': np.array([1, 3, 64, 64])},
                                 'unsq': {'unsqueeze_dims': np.array([0, 4])},
                                 'data_2': {'shape': np.array([1, 1, 3, 64, 1, 64])}
                                 })

        unsqueeze_node = Node(graph, 'unsq')
        Unsqueeze.infer(unsqueeze_node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'data_2')
        self.assertTrue(flag, resp)
