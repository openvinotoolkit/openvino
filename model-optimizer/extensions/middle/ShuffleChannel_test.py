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

from extensions.middle.ShuffleChannel import ShuffleChannel
from mo.utils.unittest.graph import build_graph_with_attrs, compare_graphs


class ShuffleChannelTests(unittest.TestCase):
    def test_1(self):
        graph = build_graph_with_attrs(
                    nodes_with_attrs=[('data', {'shape': [1, 10, 128, 128], 'kind': 'data'}),
                                      ('shuffle', {'type': 'ShuffleChannel', 'kind': 'op', 'op': 'ShuffleChannel', 'group': 2}),
                                      ('out_data', {'shape': [1, 10, 128, 128], 'kind': 'data'}),
                                      ],
                    edges_with_attrs=[('data', 'shuffle'), ('shuffle', 'out_data')]
                )
        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph_with_attrs(
                        nodes_with_attrs=[('data', {'shape': [1, 10, 128, 128], 'kind': 'data'}),
                                          ('split', {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape', 'dim': [1, 2, 5, -1]}),
                                          ('split_data', {'shape': [1, 2, 5, 128*128], 'kind': 'data'}),
                                          ('transpose', {'type': 'Permute', 'kind': 'op', 'op': 'Permute', 'order': [0, 2, 1, 3]}),
                                          ('transpose_data', {'shape': [1, 5, 2, 128*128], 'kind': 'data'}),
                                          ('concat', {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape', 'dim': [1, 10, 128, 128]}),
                                          ('out_data', {'shape': [1, 10, 128, 128], 'kind': 'data'}),
                                          ],
                        edges_with_attrs=[('data', 'split'),
                                          ('split', 'split_data'),
                                          ('split_data', 'transpose'),
                                          ('transpose', 'transpose_data'),
                                          ('transpose_data', 'concat'),
                                          ('concat', 'out_data')],
                    )
        pattern = ShuffleChannel()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'out_data', check_op_attrs=True)
        self.assertTrue(flag, resp)
