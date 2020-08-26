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
from argparse import Namespace

import numpy as np

from mo.middle.passes.mean_scale_values import move_scaleshift_to_preprocess
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    # Placeholders
                    'placeholder_1': {'value': None, 'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2': {'value': None, 'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    # ScaleShift layer
                    'scaleshift_1': {'type': 'ScaleShift', 'value': None, 'kind': 'op', 'op': 'ScaleShift'},
                    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'Result'},
                    'op_output_1': { 'kind': 'op', 'op': 'Result'},

                    'const_mul': dict(kind='op', op='Const'),
                    'const_mul_output': dict(kind='data'),
                    'mul': dict(kind='op', op='Mul'),
                    'mul_output': dict(kind='data'),

                    'const_add': dict(kind='op', op='Const'),
                    'const_add_output': dict(kind='data'),
                    'add': dict(kind='op', op='Add'),
                    'add_output': dict(kind='data'),

                    }


class TestScaleShift_To_Preprocess(unittest.TestCase):
    def test_move_scaleshift_to_preprocess_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul'),
                             ('mul', 'mul_output'),
                             ('const_mul', 'const_mul_output'),
                             ('const_mul_output', 'mul'),
                             ('mul_output', 'add'),
                             ('add', 'add_output'),
                             ('const_add', 'const_add_output'),
                             ('const_add_output', 'add'),
                             ('add_output', 'op_output'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_output': {'shape': np.array([3]), 'value': np.ones(3)},
                             'const_add_output': {'shape': np.array([3]), 'value': np.array([-1, -2, -3])},
                             })
        graph.graph['cmd_params'] = Namespace(reverse_input_channels=False)
        del graph['placeholder_1']['placeholder_1_data'][0]['in']
        #del graph['scaleshift_1']['scaleshift_1_data'][0]['in']

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'add_output'),
                                 ('add_output', 'op_output')
                                 ])

        move_scaleshift_to_preprocess(graph)
        self.assertTrue(graph.graph['mean_values'] is not None)
        self.assertTrue(np.array_equal(graph.graph['mean_values']['placeholder_1'], np.array([1, 2, 3])))

        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)

    def test_move_scaleshift_to_preprocess_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'mul'),
                             ('mul', 'mul_output'),
                             ('const_mul', 'const_mul_output'),
                             ('const_mul_output', 'mul'),
                             ('mul_output', 'add'),
                             ('add', 'add_output'),
                             ('const_add', 'const_add_output'),
                             ('const_add_output', 'add'),
                             ('add_output', 'op_output'),
                             ('placeholder_1_data', 'op_output_1')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_output': {'shape': np.array([3]), 'value': np.array((1, 2, 3))},
                             'const_add_output': {'shape': np.array([3]), 'value': np.array([-1, -2, -3])},
                             })
        graph.graph['cmd_params'] = Namespace(reverse_input_channels=False)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul'),
                                 ('mul', 'mul_output'),
                                 ('const_mul', 'const_mul_output'),
                                 ('const_mul_output', 'mul'),
                                 ('mul_output', 'add'),
                                 ('add', 'add_output'),
                                 ('const_add', 'const_add_output'),
                                 ('const_add_output', 'add'),
                                 ('add_output', 'op_output'),
                                 ('placeholder_1_data', 'op_output_1')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_output': {'shape': np.array([3]), 'value': np.array((1, 2, 3))},
                                 'const_add_output': {'shape': np.array([3]), 'value': np.array([-1, -2, -3])},
                                 })

        move_scaleshift_to_preprocess(graph)
        self.assertTrue(graph.graph.get('mean_values', None) is None)

        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)


    def test_move_scaleshift_to_preprocess_4(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'scaleshift_1'),
                             ('mul', 'mul_output'),
                             ('const_mul', 'const_mul_output'),
                             ('const_mul_output', 'mul'),
                             ('mul_output', 'add'),
                             ('add', 'add_output'),
                             ('const_add', 'const_add_output'),
                             ('const_add_output', 'add'),
                             ('add_output', 'op_output'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'const_mul_output': {'shape': np.array([3]), 'value': np.ones(3)},
                             'const_add_output': {'shape': np.array([3]), 'value': np.zeros(3)},
                             })
        graph.graph['cmd_params'] = Namespace(reverse_input_channels=False)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'add_output'),
                                 ('add_output', 'op_output')
                                 ])

        move_scaleshift_to_preprocess(graph)
        self.assertTrue(graph.graph.get('mean_values', None) is None)

        (flag, resp) = compare_graphs(graph, graph_ref, 'scaleshift_1_data')
        self.assertTrue(flag, resp)
