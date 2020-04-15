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

from extensions.middle.MinimumMiddleReplacer import MinimumMiddleReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'kind': 'data', 'data_type': None, 'value': 3, 'shape': np.array([])},

    'placeholder_2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2_data': {'value': None, 'shape': np.array([5, 5]), 'kind': 'data', 'data_type': None},

    # minimum node:
    'minimum': {'type': 'Minimum', 'kind': 'op', 'op': 'Minimum'},
    # negates
    'const_1': {'kind': 'op', 'op': 'Const', 'value': np.array(-1)},
    'const_1_d': {'kind': 'data', 'value': np.array(-1)},

    'negate_1': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    'negate_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    'const_2': {'kind': 'op', 'op': 'Const', 'value': np.array(-1)},
    'const_2_d': {'kind': 'data', 'value': np.array(-1)},

    'negate_2': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    'negate_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    'const_3': {'kind': 'op', 'op': 'Const', 'value': np.array(-1)},
    'const_3_d': {'kind': 'data', 'value': np.array(-1)},

    'negate_output': {'type': 'Multiply', 'kind': 'op', 'op': 'Mul'},
    'negate_output_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Maximum
    'maximum': {'type': 'Maximum', 'kind': 'op', 'op': 'Maximum'},
    'maximum_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # output
    'output_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
}


class MinumumMiddleReplacerTest(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_2', 'placeholder_2_data'),
                             ('placeholder_1_data', 'minimum'),
                             ('placeholder_2_data', 'minimum'),
                             ('minimum', 'output_data')
                             ])

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_2', 'placeholder_2_data'),
                                 ('placeholder_1_data', 'negate_1'),
                                 ('const_1', 'const_1_d'),
                                 ('const_1_d', 'negate_1'),
                                 ('placeholder_2_data', 'negate_2'),
                                 ('const_2', 'const_2_d'),
                                 ('const_2_d', 'negate_2'),
                                 ('negate_1', 'negate_1_data'),
                                 ('negate_2', 'negate_2_data'),
                                 ('negate_1_data', 'maximum'),
                                 ('negate_2_data', 'maximum'),
                                 ('maximum', 'maximum_data'),
                                 ('maximum_data', 'negate_output'),
                                 ('const_3', 'const_3_d'),
                                 ('const_3_d', 'negate_output'),
                                 ('negate_output', 'negate_output_data')
                                 ])

        graph.graph['layout'] = 'NHWC'

        tested_class = MinimumMiddleReplacer()
        tested_class.find_and_replace_pattern(graph=graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'minimum/negate_out_', last_node_ref='negate_output',
                                      check_op_attrs=True)
        self.assertTrue(flag, resp)
