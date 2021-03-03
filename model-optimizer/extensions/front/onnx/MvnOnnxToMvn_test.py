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

from extensions.front.onnx.MvnOnnxToMvn import MvnOnnxToMvn
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, const, connect_front

nodes = {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('mvn_onnx', {'op': 'MVNOnnx',
                                              'axes': int64_array([2, 3]),
                                              'eps': 1e-9,
                                              'eps_mode': 'outside_sqrt',
                                              'normalize_variance': 1}),
    **result(),

    # nodes after replacement
    **const('axes', int64_array([2, 3])),
    **regular_op_with_empty_data('mvn', {'op': 'MVN', 'type': None}),
}


class MvnOnnxToMvnTest(unittest.TestCase):
    def test_mvn_normalize(self):
        graph = build_graph(nodes, [('input', 'mvn_onnx'),
                                    ('mvn_onnx', 'output')],
                            nodes_with_edges_only=True)
        graph.stage = 'front'

        MvnOnnxToMvn().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes, [('input', 'mvn'),
                                        *connect_front('axes', '1:mvn'),
                                        ('mvn', 'output')],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
