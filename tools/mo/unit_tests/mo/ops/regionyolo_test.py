# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.regionyolo import RegionYoloOp
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'region': {'type': 'RegionYolo', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': { 'kind': 'op', 'op': 'Result'}
                    }


class TestRegionYOLOCaffe(unittest.TestCase):
    def test_region_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
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
                            {'node_3': {'shape': None, 'value': None},
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

    def test_region_infer_dynamic_flatten(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
                             'node_1': {'shape': shape_array([1, dynamic_dimension_value, 227, 227])},
                             'region': {'end_axis': 1, 'axis': 0, 'do_softmax': 1, **layout_attrs()}
                             })
        graph.graph['layout'] = 'NCHW'
        reorg_node = Node(graph, 'region')
        RegionYoloOp.regionyolo_infer(reorg_node)
        exp_shape = shape_array([dynamic_dimension_value, 227, 227])
        res_shape = graph.node['node_3']['shape']
        self.assertTrue(strict_compare_tensors(exp_shape, res_shape))

    def test_region_infer_flatten_again(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'region'),
                             ('region', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
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
                            {'node_3': {'shape': None, 'value': None},
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
                            {'node_3': {'shape': None, 'value': None},
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
                            {'node_3': {'shape': None, 'value': None},
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
