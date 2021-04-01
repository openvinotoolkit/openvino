"""
 Copyright (C) 2018-2021 Intel Corporation

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

from extensions.middle.PoolV2ToAttributedPool import PoolV2ToAttributedPool
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.shape import int64_array
from mo.utils.unittest.graph import build_graph, valued_const_with_data, regular_op_with_empty_data, \
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
