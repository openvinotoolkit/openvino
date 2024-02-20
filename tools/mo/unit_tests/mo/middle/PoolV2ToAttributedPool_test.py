# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.PoolV2ToAttributedPool import PoolV2ToAttributedPool
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from openvino.tools.mo.utils.shape import int64_array
from unit_tests.utils.graph import build_graph, valued_const_with_data, regular_op_with_empty_data, \
    connect, shaped_const_with_data, result


class TestPoolV2ToAttributedPool(unittest.TestCase):

    def test_pool_v2_to_attributed_pool(self):
        nodes = {
            **shaped_const_with_data('input', int64_array([200, 200])),
            **valued_const_with_data('windows', int64_array([4, 4])),
            **valued_const_with_data('strides', int64_array([4, 4])),

            **regular_op_with_empty_data('pool_v2', {'op': 'PoolingV2',
                                                     'pad': [2, 2],
                                                     'spatial_dims': [1, 2],
                                                     'auto_pad': 'same_upper',
                                                     'output_spatial_shape': [2, 3],
                                                     'pad_spatial_shape': [1, 2],
                                                     'pool_method': 'max',
                                                     'permute_attrs': None}),

            **regular_op_with_empty_data('pool_v1', {'type': 'Pooling',
                                                     'pad': [2, 2],
                                                     'spatial_dims': [1, 2],
                                                     'auto_pad': 'same_upper',
                                                     'output_spatial_shape': [2, 3],
                                                     'pad_spatial_shape': [1, 2],
                                                     'pool_method': 'max'}),

            **result('output')
        }

        edges = [
            *connect('input', 'pool_v2:0'),
            *connect('windows', 'pool_v2:1'),
            *connect('strides', 'pool_v2:2'),
            *connect('pool_v2', 'output'),
        ]

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        PoolV2ToAttributedPool().find_and_replace_pattern(graph)

        ref_graph = build_graph(nodes, [*connect('input', 'pool_v1'), *connect('pool_v1', 'output')],
                                nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, ref_graph, 'output')
        self.assertTrue(flag, resp)
