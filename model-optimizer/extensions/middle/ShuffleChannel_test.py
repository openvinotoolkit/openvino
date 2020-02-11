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

from extensions.middle.ShuffleChannel import ShuffleChannel
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate import shape_inference
from mo.utils.unittest.graph import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs


class ShuffleChannelTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(
            {'placeholder': {'kind': 'op', 'op': 'Placeholder', 'type': 'Placeholder', 'shape': [1, 10, 128, 128]},
             'data': {'shape': [1, 10, 128, 128], 'kind': 'data'},
             'shuffle': {'type': 'ShuffleChannel', 'kind': 'op', 'op': 'ShuffleChannel', 'group': 2},
             'out_data': {'shape': [1, 10, 128, 128], 'kind': 'data'},
             'output': {'kind': 'op', 'op': 'OpOutput'}
             },
            [('placeholder', 'data'), ('data', 'shuffle'), ('shuffle', 'out_data'), ('out_data', 'output')], {})
        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(
            {'placeholder': {'kind': 'op', 'op': 'Placeholder', 'type': 'Placeholder',
                             'shape': [1, 10, 128, 128]},
             'data': {'shape': [1, 10, 128, 128], 'kind': 'data'},
             'reshape': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
             'reshape_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': int64_array([1, 2, 5, -1]),
                               'shape': [4]},
             'reshape_const_data': {'kind': 'data', 'value': [1, 2, 5, -1], 'shape': [4]},
             'reshape_data': {'shape': [1, 2, 5, 128 * 128], 'kind': 'data'},
             'order_const': {'kind': 'op', 'op': 'Const', 'value': np.array([0, 2, 1, 3])},
             'order_data': {'kind': 'data', 'value': np.array([0, 2, 1, 3]), 'shape': np.array([4])},
             'transpose': {'type': 'Transpose', 'kind': 'op', 'op': 'Transpose'},
             'transpose_data': {'shape': [1, 5, 2, 128 * 128], 'kind': 'data'},
             'reshape_back': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
             'reshape_back_const': {'type': 'Const', 'kind': 'op', 'op': 'Const',
                                    'value': int64_array([1, 10, 128, 128]), 'shape': [4]},
             'reshape_back_const_data': {'kind': 'data', 'value': [1, 10, 128, 128], 'shape': [4]},
             'out_data': {'shape': [1, 10, 128, 128], 'kind': 'data'},
             'output': {'kind': 'op', 'op': 'OpOutput'},
             },
            [('placeholder', 'data'),
             ('data', 'reshape'),
             ('reshape_const', 'reshape_const_data'),
             ('reshape_const_data', 'reshape'),
             ('reshape', 'reshape_data'),
             ('order_const', 'order_data'),
             ('order_data', 'transpose', {'in': 1}),
             ('reshape_data', 'transpose', {'in': 0}),
             ('transpose', 'transpose_data'),
             ('transpose_data', 'reshape_back'),
             ('reshape_back_const', 'reshape_back_const_data'),
             ('reshape_back_const_data', 'reshape_back'),
             ('reshape_back', 'out_data'),
             ('out_data', 'output')],
            {})

        ShuffleChannel().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
