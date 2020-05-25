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

from extensions.middle.EmbeddingBagResolver import EmbeddingBagResolver
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, \
    connect


class AtenToEmbeddingBagTest(unittest.TestCase):
    def test(self):
        nodes = {
            **valued_const_with_data('weights_inp', np.random.randn(100, 2)),
            **regular_op_with_shaped_data('indices_inp', [20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('offsets_inp', [10], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('aten', [10, 2],
                                          {'type': None, 'kind': 'op', 'op': 'ATenEmbeddingBag', 'mode': 0,
                                           'name': 'my_aten'}),

            **regular_op_with_shaped_data('emb_bag', [10, 2], {'type': 'EmbeddingBagOffsetsSum', 'kind': 'op',
                                                               'op': 'EmbeddingBagOffsetsSum'}),
            **result('result'),
        }
        edges = [*connect('weights_inp', '0:aten'),
                 *connect('indices_inp', '1:aten'),
                 *connect('offsets_inp', '2:aten'),
                 *connect('aten', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        edges_ref = [*connect('weights_inp', '0:emb_bag'),
                     *connect('indices_inp', '1:emb_bag'),
                     *connect('offsets_inp', '2:emb_bag'),
                     *connect('emb_bag', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        EmbeddingBagResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_packed(self):
        nodes = {
            **valued_const_with_data('weights_inp', np.random.randn(100, 4)),
            **regular_op_with_shaped_data('indices_inp', [10, 2], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('aten', [10, 4],
                                          {'type': None, 'kind': 'op', 'op': 'ATenEmbeddingBag', 'mode': 0,
                                           'name': 'my_aten'}),

            **regular_op_with_shaped_data('emb_bag', [10, 4], {'type': 'EmbeddingBagPackedSum', 'kind': 'op',
                                                               'op': 'EmbeddingBagPackedSum'}),
            **result('result'),
        }
        edges = [*connect('weights_inp', '0:aten'),
                 *connect('indices_inp', '1:aten'),
                 *connect('aten', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        edges_ref = [*connect('weights_inp', '0:emb_bag'),
                     *connect('indices_inp', '1:emb_bag'),
                     *connect('emb_bag', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        EmbeddingBagResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_per_sample_weights(self):
        nodes = {
            **valued_const_with_data('weights_inp', np.random.randn(100, 2)),
            **regular_op_with_shaped_data('indices_inp', [20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('offsets_inp', [10], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('per_sample_weights', [20], {'type': 'Parameter'}),
            **regular_op_with_shaped_data('aten', [10, 2],
                                          {'type': None, 'kind': 'op', 'op': 'ATenEmbeddingBag', 'mode': 0,
                                           'name': 'my_aten'}),

            **regular_op_with_shaped_data('emb_bag', [10, 2], {'type': 'EmbeddingBagOffsetsSum', 'kind': 'op',
                                                               'op': 'EmbeddingBagOffsetsSum'}),
            **valued_const_with_data('zeros', np.zeros([1, 2])),
            **regular_op_with_shaped_data('concat', None, {'type': 'Concat', 'kind': 'op', 'op': 'Concat'}),
            'def_index': {'kind': 'op', 'value': int64_array(100), 'shape': None, 'type': 'Const'},
            'def_index_d': {'kind': 'data', 'value': None, 'shape': None},
            **result('result'),
        }
        edges = [*connect('weights_inp', '0:aten'),
                 *connect('indices_inp', '1:aten'),
                 *connect('offsets_inp', '2:aten'),
                 *connect('per_sample_weights', '3:aten'),
                 *connect('aten', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        edges_ref = [*connect('weights_inp', '0:concat'),
                     *connect('zeros', '1:concat'),
                     *connect('concat', '0:emb_bag'),
                     *connect('indices_inp', '1:emb_bag'),
                     *connect('offsets_inp', '2:emb_bag'),
                     *connect('def_index', '3:emb_bag'),
                     *connect('per_sample_weights', '4:emb_bag'),
                     *connect('emb_bag', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        EmbeddingBagResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
