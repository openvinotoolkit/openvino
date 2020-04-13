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

from extensions.front.caffe.ShuffleChannel import ShuffleChannel
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder': {'kind': 'op', 'op': 'Parameter', 'shape': int64_array([1, 48, 28, 28])},
    'shuffle_channel': {'kind': 'op', 'op': 'ShuffleChannel', 'group': int64_array(2), 'name': 'scname'},
    'result': {'kind': 'op', 'op': 'Result'},

    'shape': {'op': 'ShapeOf', 'kind': 'op'},
    'batch_gather': {'op': 'Gather', 'kind': 'op'},
    'batch_gather_idx': {'value': int64_array([0]), 'kind': 'op', 'type': 'Const'},
    'batch_gather_axis': {'value': int64_array(0), 'kind': 'op', 'type': 'Const'},

    'group': {'value': int64_array([2]), 'kind': 'op', 'type': 'Const'},

    'channel_gather': {'op': 'Gather', 'kind': 'op'},
    'channel_gather_idx': {'value': int64_array([1]), 'kind': 'op', 'type': 'Const'},
    'channel_gather_axis': {'value': int64_array(0), 'kind': 'op', 'type': 'Const'},

    'output_channels': {'op': 'Div', 'kind': 'op'},
    'div_group': {'value': int64_array([2]), 'kind': 'op', 'type': 'Const'},
    'convert': {'op': 'Cast', 'kind': 'op'},
    'const': {'value': int64_array([-1]), 'kind': 'op', 'type': 'Const'},
    'concat': {'op': 'Concat', 'kind': 'op'},
    'reshape_split': {'op': 'Reshape', 'kind': 'op'},
    'transpose': {'op': 'Transpose', 'kind': 'op'},
    'transpose_const': {'value': int64_array([0, 2, 1, 3]), 'kind': 'op', 'type': 'Const'},
    'reshape_concat': {'op': 'Reshape', 'kind': 'op'}
}


class ShuffleChannelTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes_attributes,
                            [
                                ('placeholder', 'shuffle_channel'),
                                ('shuffle_channel', 'result')
                            ],
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        ref_graph = build_graph(nodes_attributes,
                                [
                                    ('placeholder', 'shape', {'in': 0, 'out': 0}),

                                    ('shape', 'batch_gather', {'in': 0, 'out': 0}),
                                    ('batch_gather_idx', 'batch_gather', {'in': 1, 'out': 0}),
                                    ('batch_gather_axis', 'batch_gather', {'in': 2, 'out': 0}),

                                    ('shape', 'channel_gather', {'in': 0, 'out': 0}),
                                    ('channel_gather_idx', 'channel_gather', {'in': 1, 'out': 0}),
                                    ('channel_gather_axis', 'channel_gather', {'in': 2, 'out': 0}),

                                    ('channel_gather', 'output_channels', {'in': 0, 'out': 0}),
                                    ('div_group', 'output_channels', {'in': 1, 'out': 0}),
                                    ('output_channels', 'convert', {'in': 0, 'out': 0}),

                                    ('batch_gather', 'concat', {'in': 0, 'out': 0}),
                                    ('group', 'concat', {'in': 1, 'out': 0}),
                                    ('convert', 'concat', {'in': 2, 'out': 0}),
                                    ('const', 'concat', {'in': 3, 'out': 0}),

                                    ('placeholder', 'reshape_split', {'in': 0, 'out': 0}),
                                    ('concat', 'reshape_split', {'in': 1, 'out': 0}),

                                    ('reshape_split', 'transpose', {'in': 0, 'out': 0}),
                                    ('transpose_const', 'transpose', {'in': 1, 'out': 0}),

                                    ('transpose', 'reshape_concat', {'in': 0, 'out': 0}),
                                    ('shape', 'reshape_concat', {'in': 1, 'out': 0}),

                                    ('reshape_concat', 'result')
                                ],
                                nodes_with_edges_only=True)

        ShuffleChannel().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(Node(graph, 'result').in_port(0).get_source().node.name == 'scname')
