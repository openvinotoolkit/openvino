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

from extensions.middle.ShufflenetReshape import FeatureShuffleReshape, ReshapeSoftmaxReshape
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # Reshape layers
    'reshape_1': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape', 'dim': None},
    'reshape_1_data': {'name': 'reshape_1_data', 'value': None, 'shape': None, 'kind': 'data'},
    'reshape_2': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_2_data': {'name': 'reshape_2_data', 'value': None, 'shape': None, 'kind': 'data'},
    'reshape_3': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_3_data': {'name': 'reshape_3_data', 'value': None, 'shape': None, 'kind': 'data'},
    # Transpose layer
    'transpose_1': {'type': 'Permute', 'kind': 'op', 'op': 'Transpose'},
    'transpose_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Softmax layer
    'softmax_1': {'type': 'SoftMax', 'kind': 'op', 'op': 'SoftMax'},
    'softmax_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class FeatureShuffleReshapeTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'transpose_1'),
                             ('transpose_1', 'transpose_1_data'),
                             ('transpose_1_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 112])},
                             'reshape_1_data': {'shape': np.array([227, 227, 4, 28])},
                             'transpose_1': {'order': np.array([0, 1, 3, 2])},
                             'transpose_1_data': {'shape': np.array([227, 227, 28, 4])},
                             'reshape_2_data': {'shape': np.array([1, 227, 227, 112])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'transpose_1'),
                                 ('transpose_1', 'transpose_1_data'),
                                 ('transpose_1_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 112])},
                                 'reshape_1_data': {'shape': np.array([1, 4, 28, 227 * 227])},
                                 'transpose_1': {'order': np.array([0, 2, 1, 3])},
                                 'transpose_1_data': {'shape': np.array([1, 28, 4, 227 * 227])},
                                 'reshape_2_data': {'shape': np.array([1, 227, 227, 112])},
                                 'reshape_3_data': {'shape': np.array([1, 227, 227, 112])},
                                 })

        pattern = FeatureShuffleReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'reshape_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'transpose_1'),
                             ('transpose_1', 'transpose_1_data'),
                             ('transpose_1_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 112, 227, 227])},
                             'reshape_1_data': {'shape': np.array([1, 4, 28, 227, 227])},
                             'transpose_1': {'order': np.array([0, 2, 1, 3, 4])},
                             'transpose_1_data': {'shape': np.array([1, 28, 4, 227, 227])},
                             'reshape_2_data': {'shape': np.array([1, 112, 227, 227])},
                             })
        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'transpose_1'),
                                 ('transpose_1', 'transpose_1_data'),
                                 ('transpose_1_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 112, 227, 227])},
                                 'reshape_1_data': {'shape': np.array([1, 4, 28, 227 * 227])},
                                 'transpose_1': {'order': np.array([0, 2, 1, 3])},
                                 'transpose_1_data': {'shape': np.array([1, 28, 4, 227 * 227])},
                                 'reshape_2_data': {'shape': np.array([1, 112, 227, 227])},
                                 })

        pattern = FeatureShuffleReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'reshape_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


class ReshapeSoftmaxReshapeTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'softmax_1'),
                             ('softmax_1', 'softmax_1_data'),
                             ('softmax_1_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 2])},
                             'reshape_1': {'dim': np.array([1, 227 * 227, 2])},
                             'reshape_1_data': {'shape': np.array([1 * 227 * 227, 2])},
                             'reshape_2_data': {'shape': np.array([1, 227, 227, 2])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'softmax_1'),
                                 ('softmax_1', 'softmax_1_data'),
                                 ('softmax_1_data', 'reshape_3'),
                                 ('reshape_3', 'reshape_3_data'),
                                 ('reshape_3_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 2])},
                                 'reshape_1_data': {'shape': np.array([1, 2, 227 * 227])},
                                 'reshape_2_data': {'shape': np.array([1, 227, 227, 2])},
                                 })

        pattern = ReshapeSoftmaxReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'reshape_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'softmax_1'),
                             ('softmax_1', 'softmax_1_data'),
                             ('softmax_1_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 2])},
                             'reshape_1_data': {'shape': np.array([1 * 227 * 227, 2])},
                             'reshape_2_data': {'shape': np.array([1, 227, 227, 2])},
                             })
        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'softmax_1'),
                                 ('softmax_1', 'softmax_1_data'),
                                 ('softmax_1_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 2])},
                                 'reshape_1_data': {'shape': np.array([1 * 227 * 227, 2])},
                                 'reshape_2_data': {'shape': np.array([1, 227, 227, 2])},
                                 })

        pattern = ReshapeSoftmaxReshape()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'reshape_2_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
