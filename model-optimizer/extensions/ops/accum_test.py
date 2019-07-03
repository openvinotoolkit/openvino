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

from extensions.ops.accum import AccumOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

wrong_attrs_graph = {'node_1': {'type': 'Identity', 'kind': 'op'},
                     'accum': {'type': 'Accum', 'kind': 'op'},
                     'node_3': {'type': 'Identity', 'kind': 'op'},
                     'op_output': { 'kind': 'op', 'op': 'OpOutput'}
                     }

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_2': {'type': 'Identity', 'kind': 'op'},
                    'accum': {'type': 'Accum', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'OpOutput'}
                    }


class TestAccumOp(unittest.TestCase):
    def test_accum_infer_assertion(self):
        graph = build_graph(wrong_attrs_graph,
                            [('node_1', 'accum'),
                             ('accum', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'accum': {
                                 'top_height': 0,
                                 'top_width': 0,
                                 'size_divisible_by': 0,
                                 'have_reference': 1
                             }
                             })

        accum_node = Node(graph, 'accum')
        self.assertRaises(AssertionError, AccumOp.accum_infer, accum_node)

    def test_accum_infer_have_reference(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'accum'),
                             ('node_2', 'accum'),
                             ('accum', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([1, 3, 227, 227])},
                             'accum': {
                                 'top_height': 0,
                                 'top_width': 0,
                                 'size_divisible_by': 0,
                                 'have_reference': 1
                             }
                             })

        accum_node = Node(graph, 'accum')
        AccumOp.accum_infer(accum_node)
        exp_shape = np.array([1, 6, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_accum_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'accum'),
                             ('node_2', 'accum'),
                             ('accum', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([1, 3, 227, 227])},
                             'accum': {
                                 'top_height': 0,
                                 'top_width': 0,
                                 'size_divisible_by': 0,
                                 'have_reference': 0
                             }
                             })

        accum_node = Node(graph, 'accum')
        AccumOp.accum_infer(accum_node)
        exp_shape = np.array([1, 6, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_accum_infer_top_height_top_width(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'accum'),
                             ('node_2', 'accum'),
                             ('accum', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([1, 3, 227, 227])},
                             'accum': {
                                 'top_height': 229,
                                 'top_width': 229,
                                 'size_divisible_by': 0,
                                 'have_reference': 0
                             }
                             })

        accum_node = Node(graph, 'accum')
        AccumOp.accum_infer(accum_node)
        exp_shape = np.array([1, 6, 229, 229])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
