# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import override_placeholder_shapes, partial_infer
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_3_data': {'value': None, 'kind': 'data', 'data_type': None},
                    # Placeholders
                    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_1': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'pl_2': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
                    'pl_2_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    # ScaleShift layer
                    'scaleshift_1': {'type': 'ScaleShift', 'kind': 'op', 'op': 'ScaleShift'},
                    'scaleshift_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_b': {'value': None, 'shape': None, 'kind': 'data'},
                    'scaleshift_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    # Mul op
                    'mul_1': {'type': None, 'kind': 'op', 'op': 'Mul'},
                    'mul_1_w': {'value': None, 'shape': None, 'kind': 'data'},
                    'mul_1_data': {'value': None, 'shape': None, 'kind': 'data'},
                    'op_output': { 'kind': 'op', 'op': 'Result', 'infer': lambda x: None}
                    }


class TestInferPass(UnitTestWithMockedTelemetry):
    def test_override_placeholder_shapes(self):
        """
        Test for overriding shape in placeholder by shape from user_shapes.
        """
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Parameter'}
                             },
                            nodes_with_edges_only=True)

        ph_shape = np.array([1, 3, 224, 224])
        user_dict = {'node_1': [{'shape': ph_shape}]}
        override_placeholder_shapes(graph, user_dict)
        res_shape = graph.node['node_1']['shape']
        self.assertTrue(np.array_equal(ph_shape, res_shape))

    def test_override_placeholder_no_shape(self):
        """
        Test for case when user_shapes is not defined.
        """
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None, 'op': 'Parameter'},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Parameter'}
                             },
                            nodes_with_edges_only=True)
        out = override_placeholder_shapes(graph, None)
        res_shape = graph.node['node_1']['shape']
        placeholder_shape = np.array([1, 3, 227, 227])
        self.assertIsNone(out)
        self.assertTrue(np.array_equal(placeholder_shape, res_shape))

    def test_override_placeholder_shapes(self):
        """
        Test for case when user_shapes is not None, but it shouldn't rewrite shapes.
        """
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Parameter'}
                             },
                            nodes_with_edges_only=True)

        node_1_shape = np.array([1, 3, 227, 227])
        user_dict = {'some_node': [{'shape': np.zeros((3))}]}
        override_placeholder_shapes(graph, user_dict)
        res_shape = graph.node['node_1']['shape']
        self.assertTrue(np.array_equal(node_1_shape, res_shape))

    def test_override_placeholder_shapes_dict(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None, 'op': 'Parameter'},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Parameter'}
                             },
                            nodes_with_edges_only=True)

        placeholder_shape = np.array([1, 3, 224, 224])
        user_shapes = {
            'node_1': [{'shape': placeholder_shape}],
            'node_2': [{'shape': placeholder_shape}],
        }
        override_placeholder_shapes(graph, user_shapes)
        res_shape = graph.node['node_1']['shape']
        res_shape2 = graph.node['node_2']['shape']
        self.assertTrue(np.array_equal(placeholder_shape, res_shape))
        self.assertTrue(np.array_equal(placeholder_shape, res_shape2))

    nodes = {
        'placeholder_1': {'name': 'placeholder_1', 'shape': [1, 2, 3, 4], 'type': 'Parameter', 'value': None,
                          'kind': 'op', 'op': 'Parameter'},
        'placeholder_2': {'name': 'placeholder_2', 'shape': [5, 6, 7, 8], 'type': 'Parameter', 'value': None,
                          'kind': 'op', 'op': 'Parameter'},
        '1': {'name': 'node_1', 'type': 'Identity', 'value': None, 'kind': 'op'},
        '2': {'name': 'node_2', 'type': 'Identity', 'value': None, 'kind': 'op'},
        '3': {'name': 'concat', 'type': 'Identity', 'value': None, 'kind': 'op'},
        '4': {'name': 'output', 'type': 'SoftMax', 'value': None, 'kind': 'op'}
    }
    edges = [
        ('placeholder_1', '1'),
        ('1', '3'),
        ('placeholder_2', '2'),
        ('2', '3'),
        ('3', '4')
    ]

    def test_override_placeholder_shapes_batch_is_not_set(self):
        """
        Test case when batch is not set. (shapes shouldn't change)
        """
        graph = build_graph(self.nodes, self.edges)
        shapes = {}
        batch = None
        override_placeholder_shapes(graph, shapes, batch)
        res_shape_1 = graph.node['placeholder_1']['shape']
        res_shape_2 = graph.node['placeholder_2']['shape']
        self.assertTrue(np.array_equal(self.nodes['placeholder_1']['shape'], res_shape_1))
        self.assertTrue(np.array_equal(self.nodes['placeholder_2']['shape'], res_shape_2))

    def test_override_placeholder_shapes_real_inputs_and_batch(self):
        """
        Test case when batch is set and shapes should overwrite by user shapes.
        """
        graph = build_graph(self.nodes, self.edges)
        shapes = {'placeholder_1': [{'shape': np.array([1, 2, 3, 4])}],
                  'placeholder_2': [{'shape': np.array([1, 5, 6, 7])}]}
        batch = 4
        override_placeholder_shapes(graph, shapes, batch)
        res_shape_1 = graph.node['placeholder_1']['shape']
        res_shape_2 = graph.node['placeholder_2']['shape']
        self.assertTrue(np.array_equal(res_shape_1, np.array([4, 2, 3, 4])))
        self.assertTrue(np.array_equal(res_shape_2, np.array([4, 5, 6, 7])))

    def test_override_placeholder_shapes_real_inputs_and_batch_2(self):
        """
        Test case when batch is set, but shapes in user_shapes is None.
        """
        graph = build_graph(self.nodes, self.edges)
        shapes = {'placeholder_1': [{'shape': None}], 'placeholder_2': [{'shape': None}]}
        batch = 4
        graph.node['placeholder_2']['shape'] = np.array([1, 2, 3, 4])
        graph.node['placeholder_2']['shape'] = np.array([1, 5, 6, 7])
        override_placeholder_shapes(graph, shapes, batch)
        np.testing.assert_array_equal(graph.node['placeholder_1']['shape'], np.array([4, 2, 3, 4]))
        np.testing.assert_array_equal(graph.node['placeholder_2']['shape'], np.array([4, 5, 6, 7]))

    def test_partial_infer(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'kind': 'data', 'shape': None, 'infer': None},
                             'node_1': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'node_2': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'concat': {'kind': 'op', 'axis': 2, 'infer': concat_infer}
                             },
                            nodes_with_edges_only=True)

        start_node = 'concat'
        partial_infer(graph, start_node)
        node = Node(graph, start_node)
        self.assertTrue(node.is_partial_inferred)
        self.assertTrue(node.out_node().is_partial_inferred)

        # check if previous nodes are not inferred
        node = Node(graph, start_node)
        while True:
            # collect nodes in a list
            if isinstance(node.in_nodes(), list):
                in_nodes = node.in_nodes()
            else:
                in_nodes = [y for x, y in node.in_nodes().items()]

            # check parents and find next parent
            for n in in_nodes:
                if 'embedded_input_' not in n.id:
                    node = n
                self.assertFalse(n.has('is_partial_inferred'))

            if not len(in_nodes):
                break

    def test_partial_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2'),
                             ('node_2', 'op_output')
                             ],
                            {'node_2': {'shape': None, 'infer': None},
                             'node_1': {'shape': None, 'infer': None}
                             },
                            nodes_with_edges_only=True)
        self.assertRaises(Error, partial_infer, graph, 'node_1')

    def test_partial_infer_cycle(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'concat'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'kind': 'data', 'shape': None, 'infer': None},
                             'node_1': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'node_2': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'concat': {'kind': 'op', 'axis': 2, 'infer': concat_infer}
                             },
                            nodes_with_edges_only=True)

        start_node = 'concat'
        self.assertRaises(Error, partial_infer, graph, start_node)


class CycleTest(UnitTestWithMockedTelemetry):
    def test_is_not_fully_inferred_param(self):
        # Node that have is_not_fully_inferred=True
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'kind': 'data', 'shape': None, 'infer': None},
                             'node_1': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'node_2': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'concat': {'kind': 'op', 'axis': 2, 'infer': concat_infer, 'is_not_fully_inferred': True}
                             },
                            nodes_with_edges_only=True)

        start_node = 'concat'
        try:
            partial_infer(graph, start_node)
        except Error:
            self.fail("Unexpected Error raised")
        node = Node(graph, start_node)
        self.assertTrue(node.is_partial_inferred)
        self.assertTrue(node.out_node().is_partial_inferred)

    def test_for_is_cyclic1(self):
        # Test for case of cyclic graph without is_cyclic attrs
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_1_data'),
                             ('node_1_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('node_3_data', 'node_1')],
                            nodes_with_edges_only=True)
        with self.assertRaisesRegex(Error, 'Graph contains a cycle. Can not proceed.*'):
            partial_infer(graph)
