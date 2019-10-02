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
from argparse import Namespace

import numpy as np

from extensions.middle.ScaleInput import ScaleInput
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_3_data': {'value': None, 'kind': 'data', 'data_type': None},
                    # Placeholders
                    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'pl_2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_2_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    # ScaleShift layer
                    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
                    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    # Mul op
                    'mul_1': {'type': None, 'kind': 'op', 'op': 'Mul'},
                    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result', 'infer': lambda x: None}
                    }


class ScaleInputTests(unittest.TestCase):
    def test_scale_input_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'op_output')
                             ],
                            {'placeholder_1': {'shape': np.array([1, 3, 224, 224])}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1_data'),
                                 ('mul_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'op_output')
                                 ],
                                {'mul_1_w': {'shape': np.array([1, 1, 1]), 'value': np.array([1 / 255])}},
                                nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        graph.graph['cmd_params'] = Namespace(scale=255)
        ScaleInput().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data')
        self.assertTrue(flag, resp)

    def test_scale_input_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'op_output')
                             ],
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'op_output')
                                 ],
                                nodes_with_edges_only=True)
        graph.graph['cmd_params'] = Namespace(scale=1)
        ScaleInput().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data')
        self.assertTrue(flag, resp)
