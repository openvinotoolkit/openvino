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

from extensions.front.EmbeddingBagFuse import EmbeddingBagFuse
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, \
    regular_op, const


class EmbeddingBagFuseTest(unittest.TestCase):
    def test(self):
        nodes = {
            **regular_op('indices_inp', {'type': 'Parameter'}),
            **regular_op('offsets_inp', {'type': 'Parameter'}),
            **const('weights', np.random.randn(100, 2)),

            **regular_op('concat_before', dict(op='Concat')),
            # 1st branch
            **regular_op('1_gather_before1_1', dict(op='Gather')),
            **regular_op('1_unsqueeze_before1_1', dict(op='Unsqueeze')),
            **regular_op('1_gather_before2_1', dict(op='Gather')),
            **regular_op('1_unsqueeze_before2_1', dict(op='Unsqueeze')),
            **const('1_slice1_axes', int64_array([0])),
            **regular_op('1_slice1', dict(op='Slice')),
            **const('1_gather_after1_axis', int64_array(0)),
            **regular_op('1_gather_after1', dict(op='Gather')),
            **regular_op('1_reduce1', dict(op='ReduceSum')),
            **regular_op('1_unsqueeze_after1', dict(op='Unsqueeze')),
            # 2nd branch
            **regular_op('2_gather_before1_1', dict(op='Gather')),
            **regular_op('2_unsqueeze_before1_1', dict(op='Unsqueeze')),
            **regular_op('2_gather_before2_1', dict(op='Gather')),
            **regular_op('2_unsqueeze_before2_1', dict(op='Unsqueeze')),
            **const('2_slice1_axes', int64_array([0])),
            **regular_op('2_slice1', dict(op='Slice')),
            **const('2_gather_after1_axis', int64_array(0)),
            **regular_op('2_gather_after1', dict(op='Gather')),
            **regular_op('2_reduce1', dict(op='ReduceSum')),
            **regular_op('2_unsqueeze_after1', dict(op='Unsqueeze')),

            **regular_op('concat_after', dict(op='Concat')),

            **regular_op('emb_bag', {'type': 'EmbeddingBagOffsetsSum', 'kind': 'op', 'op': 'EmbeddingBagOffsetsSum'}),
            **result('result'),
        }
        edges = [
            ('offsets_inp', 'concat_before', {'out': 0, 'in': 0}),
            # connect 1st branch
            ('concat_before', '1_gather_before1_1'),
            ('concat_before', '1_gather_before2_1'),
            ('1_gather_before1_1', '1_unsqueeze_before1_1'),
            ('1_gather_before2_1', '1_unsqueeze_before2_1'),
            ('indices_inp', '1_slice1', {'out': 0, 'in': 0}),
            ('1_unsqueeze_before1_1', '1_slice1', {'out': 0, 'in': 1}),
            ('1_unsqueeze_before2_1', '1_slice1', {'out': 0, 'in': 2}),
            ('1_slice1_axes', '1_slice1', {'out': 0, 'in': 3}),
            ('weights', '1_gather_after1', {'out': 0, 'in': 0}),
            ('1_slice1', '1_gather_after1', {'out': 0, 'in': 1}),
            ('1_gather_after1_axis', '1_gather_after1', {'out': 0, 'in': 2}),
            ('1_gather_after1', '1_reduce1'),
            ('1_reduce1', '1_unsqueeze_after1'),
            ('1_unsqueeze_after1', 'concat_after'),
            # connect 2nd branch
            ('concat_before', '2_gather_before1_1'),
            ('concat_before', '2_gather_before2_1'),
            ('2_gather_before1_1', '2_unsqueeze_before1_1'),
            ('2_gather_before2_1', '2_unsqueeze_before2_1'),
            ('indices_inp', '2_slice1', {'out': 0, 'in': 0}),
            ('2_unsqueeze_before1_1', '2_slice1', {'out': 0, 'in': 1}),
            ('2_unsqueeze_before2_1', '2_slice1', {'out': 0, 'in': 2}),
            ('2_slice1_axes', '2_slice1', {'out': 0, 'in': 3}),
            ('weights', '2_gather_after1', {'out': 0, 'in': 0}),
            ('2_slice1', '2_gather_after1', {'out': 0, 'in': 1}),
            ('2_gather_after1_axis', '2_gather_after1', {'out': 0, 'in': 2}),
            ('2_gather_after1', '2_reduce1'),
            ('2_reduce1', '2_unsqueeze_after1'),
            ('2_unsqueeze_after1', 'concat_after'),

            ('concat_after', 'result'),
        ]
        graph = build_graph(nodes, edges)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('weights', 'emb_bag'),
                     ('indices_inp', 'emb_bag'),
                     ('offsets_inp', 'emb_bag'),
                     ('emb_bag', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        EmbeddingBagFuse().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
