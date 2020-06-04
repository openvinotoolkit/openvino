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

from extensions.front.onnx.pad_converter import ONNXPadToPad
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const

nodes_attributes = {
    'placeholder': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    **const('pads', np.array([1, 2, 3, 4], dtype=np.int64)),
    **const('value', np.array(0.5, dtype=np.float32)),
    'onnx_pad': {'type': None, 'kind': 'op', 'op': 'ONNXPad', 'name': 'my_pad', 'mode': 'constant'},
    'result': {'type': 'Result', 'value': None, 'kind': 'op', 'op': 'Result'},

    'pad': {'type': 'Pad', 'kind': 'op', 'op': 'Pad'},
    'split': {'type': 'Split', 'kind': 'op', 'op': 'Split', 'num_splits': 2},
    **const('split_axis', np.array(0, dtype=np.int32)),
}


class AttributedClampNormalizerTest(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder', 'onnx_pad', {'in': 0, 'out': 0}),
                             ('pads', 'onnx_pad', {'in': 1, 'out': 0}),
                             ('value', 'onnx_pad', {'in': 2, 'out': 0}),
                             ('onnx_pad', 'result', {'in': 0, 'out': 0}),
                             ],
                            {}, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder', 'pad', {'in': 0, 'out': 0}),
                                 ('pads', 'split', {'in': 0, 'out': 0}),
                                 ('split_axis', 'split', {'in': 1, 'out': 0}),
                                 ('split', 'pad', {'in': 1, 'out': 0}),
                                 ('split', 'pad', {'in': 2, 'out': 1}),
                                 ('value', 'pad', {'in': 3, 'out': 0}),
                                 ('pad', 'result')
                                 ],
                                {}, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        ONNXPadToPad().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Pad')[0]]['name'] == 'my_pad')
