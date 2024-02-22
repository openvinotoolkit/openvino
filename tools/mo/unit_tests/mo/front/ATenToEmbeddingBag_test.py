# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.ATenToEmbeddingBag import AtenToEmbeddingBag
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op, const


class AtenToEmbeddingBagTest(unittest.TestCase):
    def test(self):
        nodes = {
            **const('weights_inp', np.random.randn(100, 2)),
            **regular_op('indices_inp', {'type': 'Parameter'}),
            **regular_op('offsets_inp', {'type': 'Parameter'}),
            **regular_op('aten', {'type': None, 'kind': 'op', 'op': 'ATen', 'operator': 'embedding_bag', 'mode': 0,
                                  'name': 'my_aten'}),

            **regular_op('emb_bag', {'type': 'EmbeddingBagOffsetsSum', 'kind': 'op', 'op': 'EmbeddingBagOffsetsSum'}),
            **result('result'),
        }
        edges = [('weights_inp', 'aten'),
                 ('indices_inp', 'aten'),
                 ('offsets_inp', 'aten'),
                 ('aten', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('weights_inp', 'emb_bag'),
                     ('indices_inp', 'emb_bag'),
                     ('offsets_inp', 'emb_bag'),
                     ('emb_bag', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        AtenToEmbeddingBag().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_packed(self):
        nodes = {
            **const('weights_inp', np.random.randn(100, 4)),
            **regular_op('indices_inp', {'type': 'Parameter'}),
            **regular_op('aten', {'type': None, 'kind': 'op', 'op': 'ATen', 'operator': 'embedding_bag', 'mode': 0,
                                  'name': 'my_aten'}),

            **regular_op('emb_bag', {'type': 'EmbeddingBagPackedSum', 'kind': 'op',
                                     'op': 'EmbeddingBagPackedSum'}),
            **result('result'),
        }
        edges = [('weights_inp', 'aten'),
                 ('indices_inp', 'aten'),
                 ('aten', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('weights_inp', 'emb_bag'),
                     ('indices_inp', 'emb_bag'),
                     ('emb_bag', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        AtenToEmbeddingBag().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_per_sample_weights(self):
        nodes = {
            **const('weights_inp', np.random.randn(100, 2)),
            **regular_op('indices_inp', {'type': 'Parameter'}),
            **regular_op('offsets_inp', {'type': 'Parameter'}),
            **regular_op('per_sample_weights', {'type': 'Parameter'}),
            **regular_op('aten', {'type': None, 'kind': 'op', 'op': 'ATen', 'operator': 'embedding_bag', 'mode': 0,
                                  'name': 'my_aten'}),

            **regular_op('emb_bag', {'type': 'EmbeddingBagOffsetsSum', 'kind': 'op',
                                     'op': 'EmbeddingBagOffsetsSum'}),
            **regular_op('WeightsRank', {'type': None, 'kind': 'op', 'op': 'Rank'}),
            **regular_op('WeightsRank/axis', {'type': 'Add', 'kind': 'op', 'op': 'Add'}),
            **regular_op('gather1', {'type': 'Gather', 'kind': 'op', 'op': 'Gather'}),
            **regular_op('gather2', {'type': 'Gather', 'kind': 'op', 'op': 'Gather'}),
            **regular_op('WeightsShape', {'type': 'ShapeOf', 'kind': 'op', 'op': 'ShapeOf'}),
            **regular_op('Broadcast', {'type': 'Broadcast', 'kind': 'op', 'op': 'Broadcast'}),
            **regular_op('Unsqueeze', {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'}),
            **const('WeightsShape/Axis', int64_array(0)),
            **const('zero1', int64_array(0)),
            **const('zero2', int64_array(0)),
            **const('Unsqueeze/value', int64_array(0)),
            **const('Broadcast/value', int64_array(0)),
            **const('neg', int64_array(-1)),
            **regular_op('Concat', {'type': 'Concat', 'kind': 'op', 'op': 'Concat'}),
            **result('result'),
        }
        edges = [('weights_inp', 'aten'),
                 ('indices_inp', 'aten'),
                 ('offsets_inp', 'aten'),
                 ('per_sample_weights', 'aten'),
                 ('aten', 'result'),
                 ]
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('weights_inp', 'Concat', {'in': 0, 'out': 0}),
                     ('weights_inp', 'WeightsShape', {'in': 0, 'out': 0}),
                     ('weights_inp', 'WeightsRank', {'in': 0, 'out': 0}),
                     ('WeightsRank', 'WeightsRank/axis'),
                     ('neg', 'WeightsRank/axis'),
                     ('WeightsShape', 'gather1', {'in': 0, 'out': 0}),
                     ('WeightsRank/axis', 'gather1'),
                     ('WeightsShape/Axis', 'gather1'),
                     ('WeightsShape', 'gather2', {'in': 0, 'out': 0}),
                     ('zero1', 'gather2'),
                     ('zero2', 'gather2'),
                     ('Broadcast/value', 'Broadcast'),
                     ('gather1', 'Broadcast'),
                     ('Broadcast', 'Unsqueeze'),
                     ('Unsqueeze/value', 'Unsqueeze'),
                     ('Unsqueeze', 'Concat'),
                     ('Concat', 'emb_bag'),
                     ('indices_inp', 'emb_bag'),
                     ('offsets_inp', 'emb_bag'),
                     ('gather2', 'emb_bag'),
                     ('per_sample_weights', 'emb_bag'),
                     ('emb_bag', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref, nodes_with_edges_only=True)

        AtenToEmbeddingBag().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
