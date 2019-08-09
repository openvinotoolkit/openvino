"""
 Copyright (c) 2019 Intel Corporation

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

from extensions.front.mxnet.gather import GatherFrontReplacer
from mo.utils.unittest.graph import build_graph, compare_graphs
from mo.graph.graph import Node


class GatherTest(unittest.TestCase):
    def test_embedding_replace1(self):
        graph = build_graph({'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                             'embedding_const': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'type': 'Const', 'op': 'Const'},
                             'embedding': {'type': None, 'kind': 'op', 'op': 'Embedding'},
                             'last': {'type': None, 'kind': 'op', 'op': None},
                            },
                            [('placeholder_1', 'embedding', {'out': 0, 'in': 0}),
                             ('embedding_const', 'embedding', {'out': 0, 'in': 1}),
                             ('embedding', 'last')
                             ],
                            {'placeholder_1': {'shape': np.array([32,35])},
                             'embedding_const': {'shape': np.array([2000, 650]),
                                                'bias': np.array(np.random.random_integers(0, 225, (2000, 650)))},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph({'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                                 'embedding_const': {'value': None, 'shape': None, 'kind': 'op', 'data_type': None, 'type': 'Const', 'op': 'Const'},
                                 'embedding': {'type': None, 'kind': 'op', 'op': 'Gather'},
                                 'last': {'type': None, 'kind': 'op', 'op': None},
                                },
                                [
                                 ('embedding_const', 'embedding'),
                                 ('placeholder_1', 'embedding'),
                                 ('embedding', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([32,35])},
                                 'embedding_const': {'shape': np.array([2000, 650]),
                                                'bias': np.array(np.random.random_integers(0, 225, (2000, 650)))},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = GatherFrontReplacer()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)
