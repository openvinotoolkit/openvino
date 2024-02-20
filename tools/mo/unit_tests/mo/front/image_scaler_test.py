# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.image_scaler import ImageScaler
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    # ImageScaler operation
    'im_scaler': {'type': None, 'kind': 'op', 'op': 'ImageScaler'},
    'im_scaler_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    'last_data': {'value': None, 'shape': None, 'kind': 'data'},
    # Mul and Add operations
    'mul_1': {'type': None, 'value': None, 'kind': 'op', 'op': 'Mul'},
    'const_mul_1_w': {'type': None, 'value': None, 'kind': 'op', 'op': 'Const'},
    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1': {'type': None, 'value': None, 'kind': 'op', 'op': 'Add'},
    'const_add_1_w': {'type': None, 'value': None, 'kind': 'op', 'op': 'Const'},
    'add_1_w': {'value': None, 'shape': None, 'kind': 'data'},
    'add_1_data': {'value': None, 'shape': None, 'kind': 'data'},
}


class ImageScalerTest(unittest.TestCase):
    # Tests for MIDDLE stage
    # Graph with Mul and Add operations
    def test_image_scaler_test_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'im_scaler'),
                             ('im_scaler', 'im_scaler_data'),
                             ('im_scaler_data', 'last'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(2.0), 'bias': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'last')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array(2.0).shape, 'value': np.array(2.0)},
                                 'const_add_1_w': {'shape': np.array([3, 1, 1]),
                                                   'value': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'middle'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    # Graph with Add operation
    def test_image_scaler_test_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'im_scaler'),
                             ('im_scaler', 'im_scaler_data'),
                             ('im_scaler_data', 'last'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(1.0), 'bias': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'add_1'),
                                 ('const_add_1_w', 'add_1_w'),
                                 ('add_1_w', 'add_1'),
                                 ('add_1', 'add_1_data'),
                                 ('add_1_data', 'last')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_add_1_w': {'shape': np.array([3, 1, 1]),
                                                   'value': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'middle'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    # Graph with Mul operation
    def test_image_scaler_test_3(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'im_scaler'),
                             ('im_scaler', 'im_scaler_data'),
                             ('im_scaler_data', 'last'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(2.0), 'bias': np.reshape(np.array([0, 0, 0]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1_w'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'mul_1_data'),
                                 ('mul_1_data', 'last')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array(2.0).shape, 'value': np.array(2.0)},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'middle'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    # Graph without Mul and Add operations
    def test_image_scaler_test_4(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'im_scaler'),
                             ('im_scaler', 'im_scaler_data'),
                             ('im_scaler_data', 'last'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler_data': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(1.0), 'bias': np.reshape(np.array([0, 0, 0]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'last')
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 227, 227, 3])},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'middle'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    # Tests for FRONT stage
    # Graph with Mul and Add operations
    def test_image_scaler_test_5(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'im_scaler'),
                             ('im_scaler', 'last'),
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(2.0), 'bias': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1'),
                                 ('mul_1', 'add_1'),
                                 ('const_add_1_w', 'add_1'),
                                 ('add_1', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array(2.0).shape, 'value': np.array(2.0)},
                                 'const_add_1_w': {'shape': np.array([3, 1, 1]),
                                                   'value': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    # Graph with Add operation
    def test_image_scaler_test_6(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'im_scaler'),
                             ('im_scaler', 'last'),
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(1.0), 'bias': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'add_1'),
                                 ('const_add_1_w', 'add_1'),
                                 ('add_1', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'const_add_1_w': {'shape': np.array([3, 1, 1]),
                                                   'value': np.reshape(np.array([1, 2, 3]), [3, 1, 1])},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    # Graph with Mul operation
    def test_image_scaler_test_7(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'im_scaler'),
                             ('im_scaler', 'last'),
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(2.0), 'bias': np.reshape(np.array([0, 0, 0]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1'),
                                 ('const_mul_1_w', 'mul_1'),
                                 ('mul_1', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 'const_mul_1_w': {'shape': np.array(2.0).shape, 'value': np.array(2.0)},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)

    # Graph without Mul and Add operations
    def test_image_scaler_test_8(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'im_scaler'),
                             ('im_scaler', 'last'),
                             ],
                            {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                             'im_scaler': {'scale': np.array(1.0), 'bias': np.reshape(np.array([0, 0, 0]), [3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'last')
                                 ],
                                {'placeholder_1': {'shape': np.array([1, 227, 227, 3])},
                                 }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        replacer = ImageScaler()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last')
        self.assertTrue(flag, resp)
