"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.ops.resample import ResampleOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'resample': {'type': 'Resample', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'}
                    }


class TestResampleOp(unittest.TestCase):
    def test_tf_resample_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'resample'),
                             ('resample', 'node_3')],
                            {'node_3': {'is_output': True, 'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'resample': {'antialias': 1,
                                          'height': 384,
                                          'width': 512,
                                          'resample_type': 'LINEAR',
                                          'factor': 1.0}
                             })

        graph.graph['layout'] = 'NCHW'
        resample_node = Node(graph, 'resample')
        ResampleOp.resample_infer(resample_node)
        exp_shape = np.array([1, 3, 384, 512])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_factor_infer(self):
        factor = 3.0
        graph = build_graph(nodes_attributes,
                            [('node_1', 'resample'),
                             ('resample', 'node_3')],
                            {'node_3': {'is_output': True, 'shape': None},
                             'node_1': {'shape': np.array([1, 3, 224, 227])},
                             'resample': {'antialias': 1,
                                          'resample_type': 'LINEAR',
                                          'factor': factor}
                             })
        graph.graph['layout'] = 'NCHW'
        resample_node = Node(graph, 'resample')
        ResampleOp.resample_infer(resample_node)
        exp_shape = np.array([1, 3, 224 * factor, 227 * factor])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_infer(self):
        new_width = 100
        new_height = 125
        new_attrs = nodes_attributes.copy()
        new_attrs.update({'new_shape': {'value': np.array([new_height, new_width]), 'type': 'Const', 'kind': 'op'}})
        graph = build_graph(new_attrs,
                            [('node_1', 'resample'),
                             ('new_shape', 'resample'),
                             ('resample', 'node_3')],
                            {'node_3': {'is_output': True, 'shape': None},
                             'node_1': {'shape': np.array([1, 224, 227, 3])},
                             'resample': {'antialias': 1,
                                          'resample_type': 'LINEAR',
                                          'factor': 1.0,
                                          'fw': 'tf'}
                             })
        graph.graph['layout'] = 'NHWC'
        resample_node = Node(graph, 'resample')
        ResampleOp.resample_infer(resample_node)
        exp_shape = np.array([1, new_height, new_width, 3])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
