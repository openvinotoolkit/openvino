"""
 Copyright (C) 2017-2021 Intel Corporation

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

from extensions.front.caffe.MVNNormalizer import MVNCaffeToMVN
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, const, connect_front

nodes = {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('mvn_caffe', {'op': 'MVNCaffe'}),
    **result(),

    # nodes after replacement
    **const('start_1', np.array(1)),
    **const('start_2', np.array(2)),
    **const('step', np.array(1)),
    **regular_op_with_empty_data('rank', {'op': 'Rank', 'type': None}),
    **regular_op_with_empty_data('range', {'op': 'Range', 'type': None}),
    **regular_op_with_empty_data('mvn', {'op': 'MVN', 'type': None}),
}


class MVNNormalizerTest(unittest.TestCase):
    def test_mvn_normalizer(self):
        graph = build_graph(nodes, [('input', 'mvn_caffe'),
                                    ('mvn_caffe', 'output')],
                            {'mvn_caffe': {'across_channels': 0}},
                            nodes_with_edges_only=True)
        graph.stage = 'front'

        MVNCaffeToMVN().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [('input', 'mvn', {'out': 0}),
                                        ('input', 'rank', {'out': 0}),
                                        *connect_front('start_2', '0:range'),
                                        *connect_front('rank', '1:range'),
                                        *connect_front('step', '2:range'),
                                        *connect_front('range', '1:mvn'),
                                        ('mvn', 'output')],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_mvn_normalizer_across_channels(self):
        graph = build_graph(nodes, [('input', 'mvn_caffe'),
                                    ('mvn_caffe', 'output')],
                            {'mvn_caffe': {'across_channels': 1}},
                            nodes_with_edges_only=True)
        graph.stage = 'front'

        MVNCaffeToMVN().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [('input', 'mvn', {'out': 0}),
                                        ('input', 'rank', {'out': 0}),
                                        *connect_front('start_1', '0:range'),
                                        *connect_front('rank', '1:range'),
                                        *connect_front('step', '2:range'),
                                        *connect_front('range', '1:mvn'),
                                        ('mvn', 'output')],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
