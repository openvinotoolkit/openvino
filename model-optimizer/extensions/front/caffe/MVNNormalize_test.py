"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.front.caffe.MVNNormalizer import MVNCaffeFrontReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, const, connect_front


class MVNNormalizerTest(unittest.TestCase):
    def test_attributed_slice_replacer(self):
        nodes = {
            **regular_op_with_empty_data('input', {'type': 'Parameter'}),
            **regular_op_with_empty_data('mvn_caffe', {'op': 'MVNCaffe', 'across_channels': 1}),
            **result(),

            # nodes after replacement
            **const('start', np.array(1)),
            **const('step', np.array(1)),
            **regular_op_with_empty_data('rank', {'op': 'Rank', 'type': None}),
            **regular_op_with_empty_data('range', {'op': 'Range', 'type': None}),
            **regular_op_with_empty_data('mvn', {'op': 'MVN', 'type': None}),
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'mvn_caffe'),
            ('mvn_caffe', 'output'),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        MVNCaffeFrontReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'mvn', {'out': 0}),
            ('input', 'rank', {'out': 0}),
            *connect_front('start', '0:range'),
            *connect_front('rank', '1:range'),
            *connect_front('step', '2:range'),
            *connect_front('range', '1:mvn'),
            ('mvn', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
