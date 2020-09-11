"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.upsample import UpsampleOp
from generator import generator, generate
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'upsample': {'type': 'Upsample', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


@generator
class TestUpsampleOp(unittest.TestCase):
    @generate(*[
        (np.array([1., 1., 2., 2.]), np.array([1, 3, 227, 227]), np.array([1, 3, 454, 454], dtype=np.int64)),
        (np.array([1., 1., 2.5, 1.5]), np.array([1, 5, 227, 227]), np.array([1, 5, 567, 340], dtype=np.int64)),
        (np.array([1., 1., 1.3, 0.7]), np.array([1, 14, 1023, 713]), np.array([1, 14, 1329, 499], dtype=np.int64)),
    ])
    def test_upsample_with_scales_infer(self, scales, input_shape, expected_shape):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'upsample'),
                             ('upsample', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': input_shape},
                             'upsample': {'mode': 'linear',
                                          'height_scale': scales[2],
                                          'width_scale': scales[3]}
                             })

        graph.graph['layout'] = 'NCHW'
        upsample_node = Node(graph, 'upsample')
        UpsampleOp.upsample_infer(upsample_node)
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(expected_shape)):
            self.assertEqual(expected_shape[i], res_shape[i])

    @generate(*[
        (np.array([1., 1., 2., 2.]), np.array([1, 3, 227, 227]), np.array([1, 3, 454, 454], dtype=np.int64)),
        (np.array([1., 1., 2.5, 1.5]), np.array([1, 5, 227, 227]), np.array([1, 5, 567, 340], dtype=np.int64)),
        (np.array([1., 1., 1.3, 0.7]), np.array([1, 14, 1023, 713]), np.array([1, 14, 1329, 499], dtype=np.int64)),
    ])
    def test_upsample_with_second_input_infer(self, scales, input_shape, expected_shape):
        nodes_attributes['scales'] = {'kind': 'data', 'value': scales}
        graph = build_graph(nodes_attributes,
                            [('node_1', 'upsample'),
                             ('scales', 'upsample'),
                             ('upsample', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': input_shape},
                             'upsample': {'mode': 'linear',
                                          'height_scale': None,
                                          'width_scale': None}
                             })

        graph.graph['layout'] = 'NCHW'
        upsample_node = Node(graph, 'upsample')
        UpsampleOp.upsample_infer(upsample_node)
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(expected_shape)):
            self.assertEqual(expected_shape[i], res_shape[i])
