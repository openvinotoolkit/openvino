"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.ops.interp import InterpOp
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'data'},
                    'interp': {'type': 'Interp', 'kind': 'op', 'factor': None, 'parse_2nd_input': 'value'},
                    'node_3': {'type': 'Identity', 'shape': None, 'value': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'OpOutput'}
                    }


class TestInterpOp(unittest.TestCase):
    def test_caffe_interp_infer_shrink(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('interp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 1025, 2049])},
                             'interp': {'shrink_factor': 2,
                                        'height': 0,
                                        'width': 0,
                                        'zoom_factor': 1,
                                        'pad_beg': 0,
                                        'pad_end': 0}
                             })
        graph.graph['layout'] = 'NCHW'

        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 3, 513, 1025])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_interp_infer_wh(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('interp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 1024, 1, 1])},
                             'interp': {'width': 65,
                                        'height': 33,
                                        'zoom_factor': 1,
                                        'shrink_factor': 1,
                                        'pad_beg': 0,
                                        'pad_end': 0}
                             })
        graph.graph['layout'] = 'NCHW'

        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 1024, 33, 65])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_interp_infer_zoom(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('interp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 256, 33, 65])},
                             'interp': {'zoom_factor': 2,
                                        'height': 0,
                                        'width': 0,
                                        'shrink_factor': 1,
                                        'pad_beg': 0,
                                        'pad_end': 0}
                             })
        graph.graph['layout'] = 'NCHW'

        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 256, 66, 130])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_interp_infer_zoom_shrink(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('interp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 256, 33, 65])},
                             'interp': {'zoom_factor': 2,
                                        'height': 0,
                                        'width': 0,
                                        'shrink_factor': 2,
                                        'pad_beg': 0,
                                        'pad_end': 0}
                             })
        graph.graph['layout'] = 'NCHW'

        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 256, 33, 65])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_interp_infer_zoom_shrink_error(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('interp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 256, 33, 65])},
                             'interp': {'zoom_factor': 0,
                                        'height': 0,
                                        'width': 0,
                                        'shrink_factor': 0,
                                        'pad_beg': 0,
                                        'pad_end': 0}
                             })
        graph.graph['layout'] = 'NCHW'

        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        self.assertIsNone(graph.node['node_3']['shape'])

    def test_caffe_interp_infer_zoom_default(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('interp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 256, 33, 65])},
                             'interp': {'zoom_factor': 1,
                                        'height': 0,
                                        'width': 0,
                                        'shrink_factor': 1,
                                        'pad_beg': 0,
                                        'pad_end': 0
                                        }
                             })
        graph.graph['layout'] = 'NCHW'

        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 256, 33, 65])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_caffe_interp_2_blobs(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('node_2', 'interp'),
                             ('interp', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 256, 33, 66])},
                             'node_2': {'shape': np.array([1, 1, 3, 6])},
                             'interp': {'zoom_factor': 1,
                                        'shrink_factor': 1,
                                        'pad_beg': 0,
                                        'pad_end': 0,
                                        'parse_2nd_input': 'shape',
                                        }
                             })
        graph.graph['layout'] = 'NCHW'

        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 256, 3, 6])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_interp_infer_two_inputs(self):

        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('node_2', 'interp'),
                             ('interp', 'node_3')],
                            {'node_1': {'shape': np.array([1, 20, 30, 100])},
                             'node_2': {'shape': np.array([2]), 'value': np.array([2, 3])}})
        graph.graph['layout'] = 'NHWC'
        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 2, 3, 100])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_tf_interp_infer_one_input_hw(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'interp'),
                             ('interp', 'node_3')],
                            {'node_1': {'shape': np.array([1, 20, 30, 100])},
                             'interp': {'height': 4, 'width': 6, 'pad_beg': 0, 'pad_end': 0, 'zoom_factor': None,
                                        'shrink_factor': None}})
        graph.graph['layout'] = 'NHWC'
        interp_node = Node(graph, 'interp')
        InterpOp.interp_infer(interp_node)
        exp_shape = np.array([1, 4, 6, 100])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
