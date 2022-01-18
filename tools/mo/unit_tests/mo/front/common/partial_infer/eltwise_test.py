# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from generator import generator, generate

from openvino.tools.mo.front.common.partial_infer.eltwise import eltwise_infer
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'value': 2, 'kind': 'data'},
                    'node_2': {'value': 3, 'kind': 'data'},
                    'eltw_1': {'kind': 'op'},
                    'node_3': {'value': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


@generator
class TestEltwiseInfer(unittest.TestCase):
    @generate(*[
        (np.array(2), [], np.array(3), [], lambda a, b: np.multiply(a, b), np.array(6), []),
        (np.array(2), [], np.array(3), [], lambda a, b: np.maximum(a, b), np.array(3), []),
        (np.array(2), [], np.array(3), [], lambda a, b: np.add(a, b), np.array(5), []),
        (None, [1, 5], None, [1, 1], lambda a, b: np.add(a, b), None, [1, 5]),
        (None, [dynamic_dimension_value, 3], None, [1, 1], lambda a, b: np.add(a, b), None,
         [dynamic_dimension_value, 3]),
        (None, [dynamic_dimension_value, 3], None, [1, dynamic_dimension_value], lambda a, b: np.add(a, b), None,
         [dynamic_dimension_value, 3]),
        (None, [4, 5, dynamic_dimension_value, 3], None, [1, dynamic_dimension_value], lambda a, b: np.add(a, b), None,
         [4, 5, dynamic_dimension_value, 3]),
        (None, [1, 10, 20, 30], None, [dynamic_dimension_value, 10, 20, 30], lambda a, b: np.add(a, b), None,
         [dynamic_dimension_value, 10, 20, 30]),
        # dynamic value propagation
        (shape_array([dynamic_dimension_value, 5]), [2], np.array(3), [], lambda a, b: np.add(a, b),
         shape_array([dynamic_dimension_value, 8]), [2]),
        (shape_array([dynamic_dimension_value, 5]), [2], np.array([3, 7]), [], lambda a, b: np.add(a, b),
         shape_array([dynamic_dimension_value, 12]), [2]),
    ])
    def test_eltwise_infer(self, value1, shape1, value2, shape2, shape_infer, exp_value, exp_shape):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': shape_array(value1).shape if value1 is not None else shape_array(shape1),
                                        'value': value1},
                             'node_2': {'shape': shape_array(value2).shape if value2 is not None else shape_array(shape2),
                                        'value': value2}
                             })

        graph.graph['layout'] = 'NCHW'

        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, shape_infer)
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        if exp_value is not None:
            self.assertTrue(strict_compare_tensors(res_value, shape_array(exp_value)))
        self.assertTrue(strict_compare_tensors(res_shape, shape_array(exp_shape)))

    def test_eltwise_infer_none_val(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256]), 'value': None},
                             'node_2': {'shape': np.array([1, 3, 256, 256])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, lambda a, b: a * b)
        exp_shape = np.array([1, 3, 256, 256])
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        for i in range(0, len(exp_shape)):
            self.assertEqual(exp_shape[i], res_shape[i])

        self.assertIsNone(res_value)

    def test_eltwise_infer_none_min_max(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 257, 256])},
                             'node_2': {'shape': np.array([1, 3, 256, 257])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        with self.assertRaisesRegex(Error, 'Input shapes mismatch*'):
            eltwise_infer(eltwise_node)
