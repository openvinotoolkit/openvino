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
from generator import generator

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs


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
        },
        'unsq_dims_const': {
            'op': 'Const',
            'kind': 'op',
        },
        'unsq_dims': {
            'kind': 'data',
        },
        'data_2': {
            'kind': 'data',
            'shape': None,
            'value': None,
        }
    }

    def test_unsqueeze_infer(self):
        unsq_dims = np.array([0, 4])
        graph = build_graph(self.nodes_attributes,
                            [('data_1', 'unsq'),
                             ('unsq_dims_const', 'unsq_dims'),
                             ('unsq_dims', 'unsq'),
                             ('unsq', 'data_2')],
                            {'data_1': {'shape': np.array([1, 3, 64, 64])},
                             'unsq_dims': {'value': unsq_dims, 'shape': unsq_dims.shape},
                             'unsq_dims_const': {'value': unsq_dims, 'shape': unsq_dims.shape},
                             })

        graph_ref = build_graph(self.nodes_attributes,
                                [('data_1', 'unsq'),
                                 ('unsq_dims_const', 'unsq_dims'),
                                 ('unsq_dims', 'unsq'),
                                 ('unsq', 'data_2')],
                                {'data_1': {'shape': np.array([1, 3, 64, 64])},
                                 'unsq_dims': {'value': unsq_dims, 'shape': unsq_dims.shape},
                                 'unsq_dims_const': {'value': unsq_dims, 'shape': unsq_dims.shape},
                                 'data_2': {'shape': np.array([1, 1, 3, 64, 1, 64])},
                                 })

        unsqueeze_node = Node(graph, 'unsq')
        Unsqueeze.infer(unsqueeze_node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'data_2')
        self.assertTrue(flag, resp)

    def test_unsqueeze_infer_negative_indices(self):
        unsq_dims = np.array([-1])
        graph = build_graph(self.nodes_attributes,
                            [('data_1', 'unsq'),
                             ('unsq_dims_const', 'unsq_dims'),
                             ('unsq_dims', 'unsq'),
                             ('unsq', 'data_2')],
                            {'data_1': {'shape': np.array([2, 3, 64, 64])},
                             'unsq_dims': {'value': unsq_dims, 'shape': unsq_dims.shape},
                             'unsq_dims_const': {'value': unsq_dims, 'shape': unsq_dims.shape},
                             })

        graph_ref = build_graph(self.nodes_attributes,
                                [('data_1', 'unsq'),
                                 ('unsq_dims_const', 'unsq_dims'),
                                 ('unsq_dims', 'unsq'),
                                 ('unsq', 'data_2')],
                                {'data_1': {'shape': np.array([2, 3, 64, 64])},
                                 'unsq_dims': {'value': int64_array([4]), 'shape': unsq_dims.shape},
                                 'unsq_dims_const': {'value': int64_array([4]), 'shape': unsq_dims.shape},
                                 'data_2': {'shape': np.array([2, 3, 64, 64, 1])},
                                 })

        unsqueeze_node = Node(graph, 'unsq')
        Unsqueeze.infer(unsqueeze_node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'data_2')
        self.assertTrue(flag, resp)
