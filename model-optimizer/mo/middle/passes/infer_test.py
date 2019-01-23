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

from mo.front.common.partial_infer.concat import concat_infer
from mo.graph.graph import Node
from mo.middle.passes.infer import override_placeholder_shapes, partial_infer, add_mean_scale_values, scale_input, \
    check_for_cycle
from mo.utils.cli_parser import get_mean_scale_dictionary, parse_tuple_pairs
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph, compare_graphs

nodes_attributes = {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'node_2': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'concat': {'type': 'Concat', 'value': None, 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
                    'node_3_data': {'value': None, 'kind': 'data', 'data_type': None},
                    # Placeholders
                    'placeholder_1': {'shape': None, 'type': 'Input', 'kind': 'op', 'op': 'Placeholder'},
                    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
                    'placeholder_2': {'shape': None, 'type': 'Input', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_1': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
                    'pl_1_data': {'value': None, 'kind': 'data', 'data_type': None},
                    'pl_2': {'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
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
                    }


class TestInferPass(unittest.TestCase):
    def test_override_placeholder_shapes(self):
        """
        Test for overriding shape in placeholder by shape from user_shapes.
        """
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {'node_2': {'is_output': True, 'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder'}
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
                            [('node_1', 'node_2')],
                            {'node_2': {'is_output': True, 'shape': None, 'op': 'Placeholder'},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder'}
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
                            [('node_1', 'node_2')],
                            {'node_2': {'is_output': True, 'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder'}
                             },
                            nodes_with_edges_only=True)

        node_1_shape = np.array([1, 3, 227, 227])
        user_dict = {'some_node': [{'shape': np.zeros((3))}]}
        override_placeholder_shapes(graph, user_dict)
        res_shape = graph.node['node_1']['shape']
        self.assertTrue(np.array_equal(node_1_shape, res_shape))

    def test_override_placeholder_shapes_dict(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {'node_2': {'is_output': True, 'shape': None, 'op': 'Placeholder'},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder'}
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
        'placeholder_1': {'name': 'placeholder_1', 'shape': [1, 2, 3, 4], 'type': 'Placeholder', 'value': None,
                          'kind': 'op', 'op': 'Placeholder'},
        'placeholder_2': {'name': 'placeholder_2', 'shape': [5, 6, 7, 8], 'type': 'Placeholder', 'value': None,
                          'kind': 'op', 'op': 'Placeholder'},
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
                             ('concat', 'node_3')],
                            {'node_3': {'kind': 'data', 'is_output': True, 'shape': None, 'infer': None},
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
                            [('node_1', 'node_2')],
                            {'node_2': {'is_output': True, 'shape': None, 'infer': None},
                             'node_1': {'shape': None, 'infer': None}
                             },
                            nodes_with_edges_only=True)
        self.assertRaises(Error, partial_infer, graph, 'node_1')

    def test_partial_infer_cycle(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'concat')],
                            {'node_3': {'kind': 'data', 'is_output': True, 'shape': None, 'infer': None},
                             'node_1': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'node_2': {'kind': 'data', 'shape': np.array([1, 3, 227, 227]), 'infer': None},
                             'concat': {'kind': 'op', 'axis': 2, 'infer': concat_infer}
                             },
                            nodes_with_edges_only=True)

        start_node = 'concat'
        self.assertRaises(Error, partial_infer, graph, start_node)

    def test_add_mean_scale_values_with_data_name(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {'node_2': {'is_output': True, 'shape': None, 'data_type': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder', 'name': 'data',
                                        'data_type': None}
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        mean_values = parse_tuple_pairs('(124,117,104)')
        scale_values = parse_tuple_pairs('')

        # input = 'data'
        mean_scale = get_mean_scale_dictionary(mean_values, scale_values, None)
        self.assertEqual(len(graph), 2)
        add_mean_scale_values(graph, mean_scale)
        self.assertEqual(len(graph), 5)

    def test_add_mean_scale_values_without_data_name(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_2')],
                            {'node_2': {'is_output': True, 'shape': None, 'data_type': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227]), 'op': 'Placeholder', 'name': 'data',
                                        'data_type': None}
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        mean_values = parse_tuple_pairs('(124,117,104)')
        scale_values = parse_tuple_pairs('')
        # input = None
        mean_scale = get_mean_scale_dictionary(mean_values, scale_values, None)
        self.assertEqual(len(graph), 2)
        add_mean_scale_values(graph, mean_scale)
        self.assertEqual(len(graph), 5)

    def test_add_mean_scale_values1(self):
        graph = build_graph(nodes_attributes,
                            [('pl_1', 'pl_1_data'), ('pl_2', 'pl_2_data')],
                            {'pl_1_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_2_data': {'shape': np.array([1, 6]), 'infer': None},
                             'pl_1': {'shape': np.array([1,3,38,38])},
                             'pl_2': {'shape': np.array([1,6])},
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        add_mean_scale_values(graph,
                              {'pl_1': {'mean': np.array([1., 2., 3.])}, 'pl_2': {'mean': np.array([0., 0., 0.])}})
        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 0, "Found Mul op in graph")

    def test_optimize_scale_and_add_mean_values(self):
        graph = build_graph(
            nodes_attributes,
            [
                ('pl_1', 'pl_1_data')
            ],
            {
                'pl_1_data': {
                    'shape': np.array([1, 3, 38, 38]),
                    'infer': None
                },
                'pl_1': {
                    'shape': np.array([1,3,38,38])
                }
            },
            nodes_with_edges_only=True
        )
        graph.graph['layout'] = 'NCHW'
        add_mean_scale_values(graph,
                              {
                                  'pl_1': {
                                      'scale': np.array([1.]),
                                      'mean': np.array([1., 2., 3.])
                                  }
                              })
        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 0, "Found Mul op in graph")

    def test_optimize_mean_and_add_scale_values(self):
        graph = build_graph(
            nodes_attributes,
            [
                ('pl_1', 'pl_1_data')
            ],
            {
                'pl_1_data': {
                    'shape': np.array([1, 3, 38, 38]),
                    'infer': None
                },
                'pl_1': {
                    'shape': np.array([1,3,38,38])
                }
            },
            nodes_with_edges_only=True
        )
        graph.graph['layout'] = 'NCHW'
        add_mean_scale_values(graph,
                              {
                                  'pl_1': {
                                      'scale': np.array([1.43]),
                                      'mean': np.array([0., 0., 0.])
                                  }
                              })
        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 0, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 1, "Found Mul op in graph")

    def test_add_mean_scale_values3(self):
        graph = build_graph(nodes_attributes,
                            [('pl_1', 'pl_1_data')],
                            {'pl_1_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_1': {'shape': np.array([1,3,38,38])},
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        add_mean_scale_values(graph, [[np.array([1., 2., 3.]), np.array([1., 2., 3.])]])

        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "Found more than one Add op in graph")
        self.assertEqual(mul_op_cnt, 1, "Found more than one Nul op in graph")

    def test_add_mean_scale_values_cut_graph(self):
        """
        Test case when user cutted start of the network and specified mean/scale value to the new input node 'node_3'.
        """
        graph = build_graph(nodes_attributes,
                            [('pl_1', 'pl_1_data'),
                             ('pl_2', 'pl_2_data'),
                             ('pl_2_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('pl_1_data', 'node_1'),
                             ('node_3_data', 'node_1'),
                             ],
                            {'pl_1_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_2_data': {'shape': np.array([1, 3, 38, 38]), 'infer': None},
                             'pl_2': {'initial_node_name': 'node_3', 'shape': np.array([1,3,38,38])},
                             'pl_1': {'shape': np.array([1,3,38,38])},
                             },
                            nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        add_mean_scale_values(graph, {'pl_1': {'mean': np.array([1, 2, 3])}, 'node_3': {'scale': np.array([1, 2, 3])}})

        mul_op_cnt = 0
        add_op_cnt = 0
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.op == 'Mul':
                mul_op_cnt += 1
            if node.has_valid('op') and node.op == 'Add':
                add_op_cnt += 1

        self.assertEqual(add_op_cnt, 1, "There should be exactly one Add op")
        self.assertEqual(mul_op_cnt, 1, "There should be exactly one Mul op")
        self.assertEqual(Node(graph, 'pl_2').out_node().out_node().op, 'Mul', "The Mul op should be added after pl_2")
        self.assertEqual(Node(graph, 'pl_1').out_node().out_node().op, 'Add', "The Add op should be added after pl_1")


class ScaleInputTests(unittest.TestCase):
    def test_scale_input_1(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data')],
                            {'placeholder_1_data': {'is_output': True},
                             'placeholder_1': {'shape': np.array([1, 3, 224, 224])}
                            },
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'mul_1_data'),
                                 ('mul_1_data', 'mul_1'),
                                 ('mul_1_w', 'mul_1'),
                                 ('mul_1', 'placeholder_1_data')],
                                {'mul_1_w': {'shape': np.array([1, 1, 1]), 'value': np.array([1 / 255])},
                                 'placeholder_1_data': {'is_output': True}},
                                nodes_with_edges_only=True)
        graph.graph['layout'] = 'NCHW'
        scale_input(graph, 255)
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data')
        self.assertTrue(flag, resp)

    def test_scale_input_2(self):
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data')],
                            {'placeholder_1_data': {'is_output': True}},
                            nodes_with_edges_only=True)

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data')],
                                {'placeholder_1_data': {'is_output': True}},
                                nodes_with_edges_only=True)

        scale_input(graph, 1)
        (flag, resp) = compare_graphs(graph, graph_ref, 'placeholder_1_data')
        self.assertTrue(flag, resp)

    def test_check_for_cycle1(self):
        # cyclic case
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_1_data'),
                             ('node_1_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('node_3_data', 'node_1')],
                            nodes_with_edges_only=True)
        with self.assertRaisesRegex(Error, 'Graph contains a cycle. Can not proceed.*'):
            check_for_cycle(graph)

    def test_check_for_cycle2(self):
        # acyclic case
        graph = build_graph(nodes_attributes,
                            [('node_1', 'node_1_data'),
                             ('node_1_data', 'node_3'),
                             ('node_3', 'node_3_data'),
                             ('node_3_data', 'mul_1'),
                             ('mul_1_w', 'mul_1'),
                             ('mul_1', 'mul_1_data')
                             ],
                            nodes_with_edges_only=True)
        try:
            check_for_cycle(graph)
        except Error:
            self.fail("Unexpected Error raised")

    def test_is_not_fully_inferred_param(self):
        # Node that have is_not_fully_inferred=True
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3')],
                            {'node_3': {'kind': 'data', 'is_output': True, 'shape': None, 'infer': None},
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
