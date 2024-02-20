# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.embedding_bag import EmbeddingBagOffsetsSum, EmbeddingBagPackedSum, EmbeddingSegmentsSum
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect

nodes = {
    **valued_const_with_data('data', np.random.randn(3000, 8)),
    **regular_op_with_shaped_data('indices1d', [100], {'type': 'Parameter', 'value': None,
                                                       '_out_port_data_type': {0: np.int32}}),
    **regular_op_with_shaped_data('indices2d', [30, 3], {'type': 'Parameter', 'value': None,
                                                         '_out_port_data_type': {0: np.int32}}),
    **regular_op_with_shaped_data('offsets', [30], {'type': 'Parameter', 'value': None,
                                                    '_out_port_data_type': {0: np.int32}}),
    **regular_op_with_shaped_data('segment_ids', [100], {'type': 'Parameter', 'value': None,
                                                       '_out_port_data_type': {0: np.int32}}),
    **valued_const_with_data('num_segments', np.array(30, dtype=np.int32)),
    **regular_op_with_shaped_data('embedding_bag_offsets', None,
                                  {'op': 'EmbeddingBagOffsetsSum', 'type': 'EmbeddingBagOffsetsSum',
                                   'name': 'embedding_bag_offsets'}),
    **regular_op_with_shaped_data('embedding_bag_packed', None,
                                  {'op': 'EmbeddingBagPackedSum', 'type': 'EmbeddingBagPackedSum',
                                   'name': 'embedding_bag_packed'}),
    **regular_op_with_shaped_data('embedding_segments', None,
                                  {'op': 'EmbeddingSegmentsSum', 'type': 'EmbeddingSegmentsSum',
                                   'name': 'embedding_bag_packed'}),
    **result('output'),
}


class TestEmbeddingInfer(unittest.TestCase):
    def test_embedding_bag_offsets_sum(self):
        graph = build_graph(nodes, [
            *connect('data', '0:embedding_bag_offsets'),
            *connect('indices1d', '1:embedding_bag_offsets'),
            *connect('offsets', '2:embedding_bag_offsets'),
            ('embedding_bag_offsets', 'embedding_bag_offsets_d', {'out': 0}),
            ('embedding_bag_offsets_d', 'output'),
        ], nodes_with_edges_only=True)
        eb_node = Node(graph, 'embedding_bag_offsets')
        EmbeddingBagOffsetsSum.infer(eb_node)

        self.assertTrue(np.array_equal(eb_node.out_port(0).data.get_shape(), int64_array([30, 8])))

    def test_embedding_bag_packed_sum(self):
        graph = build_graph(nodes, [
            *connect('data', '0:embedding_bag_packed'),
            *connect('indices2d', '1:embedding_bag_packed'),
            ('embedding_bag_packed', 'embedding_bag_packed_d', {'out': 0}),
            ('embedding_bag_packed_d', 'output'),
        ], nodes_with_edges_only=True)
        eb_node = Node(graph, 'embedding_bag_packed')
        EmbeddingBagPackedSum.infer(eb_node)

        self.assertTrue(np.array_equal(eb_node.out_port(0).data.get_shape(), int64_array([30, 8])))

    def test_embedding_segments_sum(self):
        graph = build_graph(nodes, [
            *connect('data', '0:embedding_segments'),
            *connect('indices1d', '1:embedding_segments'),
            *connect('segment_ids', '2:embedding_segments'),
            *connect('num_segments', '3:embedding_segments'),
            ('embedding_segments', 'embedding_segments_d', {'out': 0}),
            ('embedding_segments_d', 'output'),
        ], nodes_with_edges_only=True)
        eb_node = Node(graph, 'embedding_segments')
        EmbeddingSegmentsSum.infer(eb_node)

        self.assertTrue(np.array_equal(eb_node.out_port(0).data.get_shape(), int64_array([30, 8])))
