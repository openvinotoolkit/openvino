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
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'value': None, 'kind': 'op', 'op': 'EmbeddingBag', 'scale_grad_by_freq': 0, 'mode': 0},
                    'node_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'node_2': {'value': None, 'kind': 'op', 'op': 'EmbeddingBag', 'scale_grad_by_freq': 0, 'mode': 0},
                    'node_2_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'gather_1': {'type': 'Gather', 'value': None, 'kind': 'op'},
                    'gather_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'ws_1': {'type': 'ExperimentalSparseWeightedSum', 'value': None, 'kind': 'op'},
                    'ws_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'const_axis': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None, 'shape': None},
                    'axis_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'const_default': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None, 'shape': None},
                    'default_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'const_dense_shape': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None, 'shape': None},
                    'dense_shape_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None, 'shape': None},
                    'const_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'concat_data': {'value': None, 'kind': 'data', 'data_type': None},
                    # Placeholders
                    'indices': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'indices_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'offsets': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'offsets_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'const_weights': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None, 'shape': None},
                    'weights_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

                    'op_output': {'kind': 'op', 'op': 'Result', 'infer': lambda x: None}
                    }


class EmbeddingBagResolverTests(unittest.TestCase):
    def test_embedding_bag_to_gather(self):
        graph = build_graph(nodes_attributes,
                            [('const_weights', 'weights_data'),
                             ('weights_data', 'node_1'),
                             ('indices', 'indices_data'),
                             ('indices_data', 'node_1'),
                             ('offsets', 'offsets_data'),
                             ('offsets_data', 'node_1'),
                             ('node_1', 'node_1_data'),
                             ('node_1_data', 'op_output')
                             ],
                            {'indices_data': {'shape': np.array([128])},
                             'offsets_data': {'shape': np.array([128])}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('const_weights', 'weights_data'),
                                 ('weights_data', 'gather_1'),
                                 ('indices', 'indices_data'),
                                 ('indices_data', 'gather_1'),
                                 ('const_axis', 'axis_data'),
                                 ('axis_data', 'gather_1'),
                                 ('gather_1', 'gather_1_data'),
                                 ('gather_1_data', 'op_output')
                                 ],
                                {'indices_data': {'shape': np.array([128])}},
                                nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        EmbeddingBagResolver().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output')
        self.assertTrue(flag, resp)

    def test_embedding_bag_to_single_weighted_sum(self):
        graph = build_graph(nodes_attributes,
                            [('const_weights', 'weights_data'),
                             ('weights_data', 'node_1'),
                             ('indices', 'indices_data'),
                             ('indices_data', 'node_1'),
                             ('offsets', 'offsets_data'),
                             ('offsets_data', 'node_1'),
                             ('node_1', 'node_1_data'),
                             ('node_1_data', 'op_output')
                             ],
                            {'indices_data': {'shape': np.array([128])},
                             'offsets_data': {'shape': np.array([64])},
                             'weights_data': {'shape': np.array([1024, 16])}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('const_weights', 'weights_data'),
                                 ('const', 'const_data'),
                                 ('weights_data', 'concat'),
                                 ('const_data', 'concat'),
                                 ('concat', 'concat_data'),
                                 ('offsets', 'offsets_data'),
                                 ('indices', 'indices_data'),
                                 ('const_default', 'default_data'),
                                 ('const_dense_shape', 'dense_shape_data'),
                                 ('offsets_data', 'ws_1'),
                                 ('indices_data', 'ws_1'),
                                 ('dense_shape_data', 'ws_1'),
                                 ('concat_data', 'ws_1'),
                                 ('default_data', 'ws_1'),
                                 ('ws_1', 'ws_1_data'),
                                 ('ws_1_data', 'op_output')
                                 ],
                                {'indices_data': {'shape': np.array([128])},
                                 'offsets': {'shape': np.array([128, 2])}},
                                nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        EmbeddingBagResolver().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'op_output')
        self.assertTrue(flag, resp)
