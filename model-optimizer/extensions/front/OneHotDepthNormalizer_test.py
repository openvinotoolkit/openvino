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

from extensions.front.OneHotDepthNormalizer import OneHotDepthNormalizer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, \
    regular_op, const


class OneHotDepthNormalizerTest(unittest.TestCase):
    def test(self):
        nodes = {
            **regular_op('input', {'type': 'Parameter'}),
            **const('depth', int64_array([2])),
            **regular_op('onehot', {'type': 'OneHot', 'kind': 'op', 'op': 'OneHot'}),

            **regular_op('reshape', {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'}),
            **const('reshape_dims', int64_array([])),
            **result('result'),
        }
        edges = [('input', 'onehot'),
                 ('depth', 'onehot'),
                 ('onehot', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('input', 'onehot'),
                     ('depth', 'reshape'),
                     ('reshape_dims', 'reshape'),
                     ('reshape', 'onehot'),
                     ('onehot', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        OneHotDepthNormalizer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
