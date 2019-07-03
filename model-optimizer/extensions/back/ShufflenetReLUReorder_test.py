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

from extensions.back.ShufflenetReLUReorder import ShufflenetReLUReorder
from mo.utils.unittest.graph import build_graph, compare_graphs

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ReLU
    'relu_1': {'type': 'ReLU', 'kind': 'op', 'op': 'ReLU'},
    'relu_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Reshape layers
    'reshape_1': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_2': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_3': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_3_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Transpose layer
    'transpose_1': {'type': 'Permute', 'kind': 'op', 'op': 'Transpose'},
    'transpose_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Conv layer
    'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'Conv2d'},
    'conv_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class ShufflenetReLUReorderTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'relu_1'),
                             ('relu_1', 'relu_1_data'),
                             ('relu_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'transpose_1'),
                             ('transpose_1', 'transpose_1_data'),
                             ('transpose_1_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data'),
                             ('reshape_2_data', 'conv_1'),
                             ('conv_1', 'conv_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 112])},
                             'relu_1_data': {'shape': np.array([1, 227, 227, 112])},
                             'reshape_1_data': {'shape': np.array([227, 227, 4, 28])},
                             'transpose_1': {'order': np.array([0, 1, 3, 2])},
                             'transpose_1_data': {'shape': np.array([227, 227, 28, 4])},
                             'reshape_2_data': {'shape': np.array([1, 227, 227, 112])},
                             'conv_1_data': {'shape': np.array([1, 227, 227, 112])},
                             'conv_1': {'pad': np.array([1, 1])}
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'transpose_1'),
                                 ('transpose_1', 'transpose_1_data'),
                                 ('transpose_1_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'relu_1'),
                                 ('relu_1', 'relu_1_data'),
                                 ('relu_1_data', 'conv_1'),
                                 ('conv_1', 'conv_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 112])},
                                 'relu_1_data': {'shape': np.array([1, 227, 227, 112])},
                                 'reshape_1_data': {'shape': np.array([227, 227, 4, 28])},
                                 'transpose_1': {'order': np.array([0, 1, 3, 2])},
                                 'transpose_1_data': {'shape': np.array([227, 227, 28, 4])},
                                 'reshape_2_data': {'shape': np.array([1, 227, 227, 112])},
                                 'conv_1_data': {'shape': np.array([1, 227, 227, 112])},
                                 })

        pattern = ShufflenetReLUReorder()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_2_neg(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'transpose_1'),
                             ('transpose_1', 'transpose_1_data'),
                             ('transpose_1_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data'),
                             ('reshape_2_data', 'conv_1'),
                             ('conv_1', 'conv_1_data')
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 112])},
                             'relu_1_data': {'shape': np.array([1, 227, 227, 112])},
                             'reshape_1_data': {'shape': np.array([227, 227, 4, 28])},
                             'transpose_1': {'order': np.array([0, 1, 3, 2])},
                             'transpose_1_data': {'shape': np.array([227, 227, 28, 4])},
                             'reshape_2_data': {'shape': np.array([1, 227, 227, 112])},
                             'conv_1_data': {'shape': np.array([1, 227, 227, 112])},
                             })
        graph.graph['layout'] = 'NHWC'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'transpose_1'),
                                 ('transpose_1', 'transpose_1_data'),
                                 ('transpose_1_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'conv_1'),
                                 ('conv_1', 'conv_1_data')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 112])},
                                 'relu_1_data': {'shape': np.array([1, 227, 227, 112])},
                                 'reshape_1_data': {'shape': np.array([227, 227, 4, 28])},
                                 'transpose_1': {'order': np.array([0, 1, 3, 2])},
                                 'transpose_1_data': {'shape': np.array([227, 227, 28, 4])},
                                 'reshape_2_data': {'shape': np.array([1, 227, 227, 112])},
                                 'conv_1_data': {'shape': np.array([1, 227, 227, 112])},
                                 })

        pattern = ShufflenetReLUReorder()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'conv_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
