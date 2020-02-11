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
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'upsample': {'type': 'Upsample', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


class TestUpsampleOp(unittest.TestCase):
    def test_upsample_with_scales_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'upsample'),
                             ('upsample', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'upsample': {'mode': 'linear',
                                          'height_scale': 2.,
                                          'width_scale': 2.}
                             })

        graph.graph['layout'] = 'NCHW'
        upsample_node = Node(graph, 'upsample')
        UpsampleOp.upsample_infer(upsample_node)
        exp_shape = np.array([1, 3, 454, 454])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_upsample_with_second_input_infer(self):
        nodes_attributes['scales'] = {'kind': 'data', 'value': np.array([1., 1., 2., 2.])}
        graph = build_graph(nodes_attributes,
                            [('node_1', 'upsample'),
                             ('scales', 'upsample'),
                             ('upsample', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'upsample': {'mode': 'linear',
                                          'height_scale': None,
                                          'width_scale': None}
                             })

        graph.graph['layout'] = 'NCHW'
        upsample_node = Node(graph, 'upsample')
        UpsampleOp.upsample_infer(upsample_node)
        exp_shape = np.array([1, 3, 454, 454])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
