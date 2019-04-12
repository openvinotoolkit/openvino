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

from extensions.middle.FusePermutesSequence import FusePermutesSequence
from extensions.middle.NormalizeFullyConnected import NormalizeFullyConnected
from mo.middle.passes.eliminate_test import build_graph
from mo.middle.passes.fusing.fuse_linear_ops_test import compare_graphs

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    'placeholder_1': {'name': 'placeholder_1', 'value': None, 'shape': None, 'type': 'Placeholder', 'kind': 'op',
                      'op': 'Placeholder'},
    'placeholder_1_data': {'name': 'placeholder_1_data', 'value': None, 'shape': None, 'kind': 'data',
                           'data_type': None},
    'reshape_1': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'fc': {'type': 'FullyConnected', 'value': None, 'kind': 'op', 'op': 'MatMul'},
    'fc_data': {'value': None, 'shape': None, 'kind': 'data'},
    'fc_weights': {'value': None, 'shape': None, 'kind': 'data'},

    'reshape_2': {'type': 'Reshape', 'value': None, 'kind': 'op', 'op': 'Reshape'},
    'reshape_2_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class NormalizeFullyConnectedTest(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'fc'),
                             ('fc_weights', 'fc'),
                             ('fc', 'fc_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 16, 512])},
                             'fc': {'out-size': 101},
                             'fc_weights': {'shape': np.array([512,101]), 'value': np.ones([512, 101]), 'input_channel_dim': 1},
                             'fc_data': {'shape': np.array([1, 16, 101])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'fc'),
                             ('fc_weights', 'fc'),
                             ('fc', 'fc_data'),
                             ('fc_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 16, 512])},
                             'reshape_1_data': {'shape': np.array([16, 512])},
                             'reshape_2_data': {'shape': np.array([1, 16, 101])},
                             'fc_weights': {'shape': np.array([512,101]), 'value': np.ones([512, 101])},
                             'fc': {'out-size': 101},
                             'fc_data': {'shape': np.array([16, 101])},
                             }, nodes_with_edges_only=True)

        pattern = NormalizeFullyConnected()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data', 'placeholder_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)


    def test_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'fc'),
                             ('fc_weights', 'fc'),
                             ('fc', 'fc_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([2, 32, 16, 512])},
                             'fc': {'out-size': 101},
                             'fc_weights': {'shape': np.array([512,101]), 'value': np.ones([512, 101]), 'input_channel_dim': 1},
                             'fc_data': {'shape': np.array([2, 32, 16, 101])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reshape_1'),
                             ('reshape_1', 'reshape_1_data'),
                             ('reshape_1_data', 'fc'),
                             ('fc_weights', 'fc'),
                             ('fc', 'fc_data'),
                             ('fc_data', 'reshape_2'),
                             ('reshape_2', 'reshape_2_data'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([2, 32, 16, 512])},
                             'reshape_1_data': {'shape': np.array([2 * 32 * 16, 512])},
                             'reshape_2_data': {'shape': np.array([2, 32, 16, 101])},
                             'fc_weights': {'shape': np.array([512,101]), 'value': np.ones([512, 101])},
                             'fc': {'out-size': 101},
                             'fc_data': {'shape': np.array([2 * 32 * 16, 101])},
                             }, nodes_with_edges_only=True)

        pattern = NormalizeFullyConnected()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data', 'placeholder_1_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
