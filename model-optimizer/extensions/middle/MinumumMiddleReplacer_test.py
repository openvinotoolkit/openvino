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

from extensions.middle.MinimumMiddleReplacer import MinimumMiddleReplacer
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {
    'placeholder_1': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    'placeholder_2': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # minimum node:
    'minimum': {'type': 'Minimum', 'kind': 'op', 'op': 'Minimum'},
    # negates
    'negate_1': {'type': 'Power', 'kind': 'op', 'op': 'Power', 'power': 1, 'scale': -1, 'shift': 0},
    'negate_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    'negate_2': {'type': 'Power', 'kind': 'op', 'op': 'Power', 'power': 1, 'scale': -1, 'shift': 0},
    'negate_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    'negate_output': {'type': 'Power', 'kind': 'op', 'op': 'Power', 'power': 1, 'scale': -1, 'shift': 0},
    'negate_output_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Maximum
    'maximum': {'type': 'Eltwise', 'kind': 'op', 'op': 'Max'},
    'maximum_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # output
    'output_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
}


class MinumumMiddleReplacer_test(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'minimum'),
                             ('placeholder_2_data', 'minimum'),
                             ('minimum', 'output_data')
                             ],
                            {'placeholder_1_data': {'value': 3, 'shape': np.array([])},
                             'placeholder_2_data': {'value': None, 'shape': np.array([5, 5])},
                             })

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_1_data', 'negate_1'),
                                 ('placeholder_2_data', 'negate_2'),
                                 ('negate_1', 'negate_1_data'),
                                 ('negate_2', 'negate_2_data'),
                                 ('negate_1_data', 'maximum'),
                                 ('negate_2_data', 'maximum'),
                                 ('maximum', 'maximum_data'),
                                 ('maximum_data', 'negate_output'),
                                 ('negate_output', 'negate_output_data')
                                 ])

        graph.graph['layout'] = 'NHWC'

        tested_class = MinimumMiddleReplacer()
        tested_class.find_and_replace_pattern(graph=graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'minimum/negate_out_', last_node_ref='negate_output',
                                      check_op_attrs=True)
        self.assertTrue(flag, resp)
