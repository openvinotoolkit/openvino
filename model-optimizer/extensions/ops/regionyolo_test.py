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

from extensions.ops.regionyolo import RegionYoloOp
from mo.front.common.extractors.utils import layout_attrs
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'region': {'type': 'RegionYolo', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'OpOutput'}
                    }


class TestRegionYOLOCaffe(unittest.TestCase):
    def test_region_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'region': {'axis': 1, 'end_axis': -1, 'do_softmax': 1, **layout_attrs()}
                             })
        graph.graph['layout'] = 'NCHW'
        reorg_node = Node(graph, 'region')
        RegionYoloOp.regionyolo_infer(reorg_node)
        exp_shape = np.array([1, 3 * 227 * 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_region_infer_flatten(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'region': {'end_axis': 1, 'axis': 0, 'do_softmax': 1, **layout_attrs()}
                             })
        graph.graph['layout'] = 'NCHW'
        reorg_node = Node(graph, 'region')
        RegionYoloOp.regionyolo_infer(reorg_node)
        exp_shape = np.array([1 * 3, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_region_infer_flatten_again(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'region': {'end_axis': 2, 'axis': 0, 'do_softmax': 1, **layout_attrs()}
                             })
        graph.graph['layout'] = 'NCHW'
        reorg_node = Node(graph, 'region')
        RegionYoloOp.regionyolo_infer(reorg_node)
        exp_shape = np.array([1 * 3 * 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_region_infer_do_softmax(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'region': {'do_softmax': 0, 'end_axis': -1, 'axis': 1, 'classes': 80, 'coords': 4,
                                        'mask': np.array([6, 7, 8]), **layout_attrs()}
                             })

        graph.graph['layout'] = 'NCHW'
        reorg_node = Node(graph, 'region')
        RegionYoloOp.regionyolo_infer(reorg_node)
        exp_shape = np.array([1, (80 + 4 + 1) * 3, 227, 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])


class TestRegionYOLOTF(unittest.TestCase):
    def test_region_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 227, 227, 3])},
                             'region': {'axis': 1, 'end_axis': -1, 'do_softmax': 1, **layout_attrs()}
                             })
        graph.graph['layout'] = 'NHWC'
        reorg_node = Node(graph, 'region')
        RegionYoloOp.regionyolo_infer(reorg_node)
        exp_shape = np.array([1, 3 * 227 * 227])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

    def test_region_infer_do_softmax(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 227, 227, 3])},
                             'region': {'do_softmax': 0, 'end_axis': -1, 'axis': 1, 'classes': 80, 'coords': 4,
                                        'mask': np.array([6, 7, 8]), **layout_attrs()}
                             })

        graph.graph['layout'] = 'NHWC'
        reorg_node = Node(graph, 'region')
        RegionYoloOp.regionyolo_infer(reorg_node)
        exp_shape = np.array([1, 227, 227, (80 + 4 + 1) * 3])
        res_shape = graph.node['node_3']['shape']
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])
