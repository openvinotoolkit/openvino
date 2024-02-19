# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest
from openvino.tools.mo.front.common.partial_infer.utils import strict_compare_tensors
from openvino.tools.mo.front.extractor import input_user_data_repack, output_user_data_repack, update_ie_fields, add_input_op, \
    get_node_id_with_ports
from openvino.tools.mo.front.extractor import spatial_attr_getter, add_input_ops, attr_getter, CaffePythonFrontExtractorOp, \
    add_output_ops, bool_to_str
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import build_graph, build_graph_with_edge_attrs, build_graph_with_attrs
from openvino.runtime import PartialShape


class FakePythonParam:
    def __init__(self, param: FakeMultiParam):
        self.__setattr__('python_param', param)


nodes_attributes = {'input': {'kind': 'data'},
                    'pool_1': {'type': 'Pooling', 'kind': 'op'},
                    'output': {'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


class UpdateIEFieldsTest(unittest.TestCase):
    def test_default_update_ie_fields(self):
        update_ie_fields({}, ir_version=None)

    def test_not_set_update_ie_fields(self):
        with self.assertRaisesRegex(Error, 'Unrecognized IR version.*'):
            update_ie_fields({}, ir_version='abracadabra')


class TestExtractor(unittest.TestCase):
    def test_spatial_attr_getter(self):
        input_shape = np.array([1, 125, 13, 13])
        params = {
            'kernel': np.array([1, 1, 1, 2]),
            'pad': np.array([1, 1, 3, 4]),
            'stride': np.array([1, 1, 2, 3]),
        }
        graph = build_graph(nodes_attributes,
                            [('input', 'pool_1'),
                             ('pool_1', 'output'),
                             ('output', 'op_output')
                             ],
                            {'input': {'shape': input_shape},
                             'pool_1': {**params, 'spatial_dims': [2, 3]},
                             'output': {'shape': None}})
        pool_1_node = Node(graph, 'pool_1')
        for param in params.keys():
            if type(params[param]) is np.ndarray:
                port_lambda = lambda x: x
                self.assertEqual(params[param][2],
                                 spatial_attr_getter(pool_1_node, field=param, dim=0, post=port_lambda))
                self.assertEqual(params[param][3],
                                 spatial_attr_getter(pool_1_node, field=param, dim=1, post=port_lambda))

    def test_attr_getter(self):
        nodes = {'input': {'kind': 'data'},
                 'reshape': {'type': 'Reshape', 'kind': 'op'},
                 'output': {'kind': 'data'},
                 'op_output': {'type': 'Result', 'kind': 'op'},
                 }
        input_shape = np.array([1, 125, 13, 13])
        params = {
            'dim': [1, 1, 2, 3],
            'max_size': np.array([3, 2, 1, 0])
        }
        expect_params = {
            'dim': "1,1,2,3",
            'max_size': "3,2,1,0",
        }
        graph = build_graph(nodes,
                            [('input', 'reshape'),
                             ('reshape', 'output'),
                             ('output', 'op_output')
                             ],
                            {'input': {'shape': input_shape},
                             'reshape': {**params, 'spatial_dims': [2, 3]},
                             'output': {'shape': None}})
        pool_1_node = Node(graph, 'reshape')
        for param in params.keys():
            if type(params[param]) is list:
                self.assertEqual(expect_params[param],
                                 attr_getter(pool_1_node, param))


class TestAddInputOp(unittest.TestCase):
    nodes = [
        ('op_node', {'kind': 'op'}),
        ('future_input', {'kind': 'op'}),
        ('another_node', {'kind': 'op'}),
    ]
    edges = [('future_input', 'op_node', {'in': 1, 'out': 0}),
             ('another_node', 'op_node', {'in': 0, 'out': 0})]

    def test_in_port_no_data(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges)
        new_input_shape = np.array([1, 2, 3, 4])
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges[1:],
                                           new_nodes_with_attrs=[('input_node', {'kind': 'op', 'op': 'Parameter',
                                                                                 'shape': new_input_shape})],
                                           new_edges_with_attrs=[('input_node', 'op_node', {'in': 1, 'out': 0})])
        add_input_op(graph, 'op_node', 1, data=False, shape=new_input_shape)
        graph.remove_edge('future_input', 'op_node')
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='op_node')
        self.assertTrue(flag, resp)

    def test_in_port_with_data(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges)
        graph.stage = 'middle'
        new_input_shape = np.array([1, 2, 3, 4])
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes, edges_with_attrs=self.edges[1:],
                                           new_nodes_with_attrs=[('input_node', {'kind': 'op', 'op': 'Parameter',
                                                                                 'shape': new_input_shape}),
                                                                 ('input_data', {'kind': 'data'})],
                                           new_edges_with_attrs=[('input_node', 'input_data', {'in': 0, 'out': 0}),
                                                                 ('input_data', 'op_node', {'in': 1, 'out': 0})])
        add_input_op(graph, 'op_node', 1, data=True, shape=new_input_shape)
        graph.remove_edge('future_input', 'op_node')
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='op_node')
        self.assertTrue(flag, resp)

    nodes_out = [
        ('op_node', {'kind': 'op'}),
        ('future_input', {'kind': 'op'}),
        ('another_node', {'kind': 'op'}),
    ]
    edges_out = [('op_node', 'future_input', {'in': 0, 'out': 1}),
                 ('op_node', 'another_node', {'in': 0, 'out': 0})]

    def test_out_port_no_data(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes_out, edges_with_attrs=self.edges_out)
        new_input_shape = np.array([1, 2, 3, 4])
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes_out, edges_with_attrs=self.edges_out[1:],
                                           new_nodes_with_attrs=[('input_node', {'kind': 'op', 'op': 'Parameter',
                                                                                 'shape': new_input_shape})],
                                           new_edges_with_attrs=[('input_node', 'future_input', {'in': 0, 'out': 0})])
        add_input_op(graph, 'op_node', 1, data=False, shape=new_input_shape, is_out_port=True)
        graph.remove_edge('op_node', 'future_input')
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='another_node')
        self.assertTrue(flag, resp)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='future_input')
        self.assertTrue(flag, resp)

    def test_out_port_with_data(self):
        graph = build_graph_with_attrs(nodes_with_attrs=self.nodes_out, edges_with_attrs=self.edges_out[1:],
                                       new_nodes_with_attrs=[('input_data', {'kind': 'data', 'shape': None, 'value': None})],
                                       new_edges_with_attrs=[('op_node', 'input_data', {'out': 1, 'in': 0}),
                                                             ('input_data', 'future_input', {'in': 0, 'out': 0})])
        graph.stage = 'middle'
        new_input_shape = np.array([1, 2, 3, 4])
        graph_ref = build_graph_with_attrs(nodes_with_attrs=self.nodes_out, edges_with_attrs=self.edges_out[1:],
                                           new_nodes_with_attrs=[('input_node', {'kind': 'op', 'op': 'Parameter',
                                                                                 'shape': new_input_shape}),
                                                                 ('input_data', {'kind': 'data', 'shape': None})],
                                           new_edges_with_attrs=[('input_node', 'input_data', {'in': 0, 'out': 0}),
                                                                 ('input_data', 'future_input', {'in': 0, 'out': 0})])
        add_input_op(graph, 'op_node', 1, data=True, shape=new_input_shape, is_out_port=True)
        graph.remove_edge('op_node', 'input_data')

        (flag, resp) = compare_graphs(graph, graph_ref, last_node='another_node')
        self.assertTrue(flag, resp)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='future_input')
        self.assertTrue(flag, resp)


class TestInputAddition(UnitTestWithMockedTelemetry):
    # Tests for input
    nodes = {'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
             'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder'},
             'relu_1': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder'},
             }
    edges = [
        ('node_1', 'conv_1'),
        ('conv_1', 'relu_1'),
    ]

    def test_none_out_port_raise(self):
        graph = build_graph(self.nodes, self.edges)
        shape = np.array([1, 2, 3, 4])
        inputs = {'conv_1': [{'shape': shape, 'out': None}]}
        with self.assertRaisesRegex(Error, 'Output port for input node conv_1 should be specified, it cannot be None!'):
            add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=True)

    def test_wrong_output_port_raise(self):
        graph = build_graph(self.nodes, self.edges)
        shape = np.array([1, 2, 3, 4])
        inputs = {'conv_1': [{'shape': shape, 'out': 5}]}
        with self.assertRaisesRegex(Error, 'Output port index 5 is out of number of available output ports for node'):
            add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=True)

    def test_wrong_input_port_raise(self):
        graph = build_graph(self.nodes, self.edges)
        shape = np.array([1, 2, 3, 4])
        inputs = {'conv_1': [{'shape': shape, 'in': 5}]}
        with self.assertRaisesRegex(Error, 'Input port index 5 is out of number of available input ports for node'):
            add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=True)

    def test_one_input_one_shape(self):
        shape = np.array([1, 2, 3, 4])
        inputs = {'conv_1': [{'shape': shape}]}
        nodes = {
            'old_input': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
            'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder'},
            'relu_1': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder'},
            'output': {'type': 'SoftMax', 'kind': 'op', 'op': 'NotPlaceholder'}
        }
        edges = [
            ('old_input', 'conv_1'),
            ('conv_1', 'relu_1'),
            ('relu_1', 'output')
        ]
        graph = build_graph(nodes, edges)
        add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=True)
        new_input = list(graph.in_edges('conv_1'))[0][0]
        self.assertFalse(graph.node['old_input']['is_input'])
        self.assertTrue(graph.node[new_input]['is_input'])
        self.assertTrue((new_input, 'conv_1') in graph.edges())
        self.assertTrue(('old_input', 'conv_1') not in graph.edges())
        shapes_are_equal = np.array_equal(graph.node[new_input]['shape'], shape)
        self.assertTrue(shapes_are_equal)

    def test_one_input_no_shape(self):
        shape = None
        inputs = {'conv_1': [{'shape': shape}]}
        nodes = {
            'old_input': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'old_input_data': {'kind': 'data', 'value': None, 'shape': np.array([-1, 224, 224, 3])},
            'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder'},
            'conv_1_data': {'kind': 'data', 'value': True, 'shape': np.array([-1, 224, 224, 3])},
            'relu_1': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder'},
            'relu_1_data': {'kind': 'data', 'value': None, 'shape': np.array([-1, 112, 112, 64])},
            'output': {'type': 'SoftMax', 'kind': 'op', 'op': 'NotPlaceholder'},
            'output_data': {'name': 'output_data', 'kind': 'data', 'shape': np.array([-1, 112, 112, 64])},
            'op_output': {'kind': 'op', 'op': 'Result'}
        }
        edges = [
            ('old_input', 'old_input_data'),
            ('old_input_data', 'conv_1'),
            ('conv_1', 'conv_1_data'),
            ('conv_1_data', 'relu_1'),
            ('relu_1', 'relu_1_data'),
            ('relu_1_data', 'output'),
            ('output', 'output_data'),
            ('output_data', 'op_output')
        ]
        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=False)
        new_input = list(graph.in_edges(list(graph.in_edges('conv_1'))[0][0]))[0][0]
        new_input_data = list(graph.in_edges('conv_1'))[0][0]
        self.assertFalse(graph.node['old_input']['is_input'])
        self.assertTrue(graph.node[new_input]['is_input'])
        self.assertTrue((new_input_data, 'conv_1') in graph.edges())
        self.assertTrue(('old_input_data', 'conv_1') not in graph.edges())
        self.assertIsNotNone(graph.node[new_input_data]['shape'])

    def test_two_inputs_two_shapes_positive_1(self):
        shape_1 = [1, 2, 3, 4]
        shape_2 = [4, 3, 2, 1]
        inputs = {'node_1': [{'shape': shape_1}], 'node_4': [{'shape': shape_2}]}
        nodes = {
            'input_1': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
            'input_2': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
            'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'node_2': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'node_3': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'node_4': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'output': {'kind': 'op', 'op': 'Result'}
        }
        edges = [
            ('input_1', 'node_1'),
            ('node_1', 'node_2'),
            ('node_3', 'output'),
            ('input_2', 'node_4'),
            ('node_4', 'output')
        ]
        graph = build_graph(nodes, edges)
        add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=True)
        new_input_1 = list(graph.in_edges('node_1'))[0][0]
        new_input_2 = list(graph.in_edges('node_4'))[0][0]
        self.assertFalse(graph.node['input_1']['is_input'])
        self.assertTrue(graph.node[new_input_1]['is_input'])
        self.assertTrue(graph.node[new_input_2]['is_input'])
        self.assertTrue((new_input_1, 'node_1') in graph.edges())
        self.assertTrue((new_input_2, 'node_4') in graph.edges())
        self.assertTrue(strict_compare_tensors(shape_1, graph.node[new_input_1]['shape']))
        self.assertTrue(strict_compare_tensors(shape_2, graph.node[new_input_2]['shape']))

    def test_two_inputs_two_shapes_not_all_inputs(self):
        shape_1 = [1, 2, 3, 4]
        shape_2 = [4, 3, 2, 1]
        inputs = {'node_1': [{'shape': shape_1}], 'node_4': [{'shape': shape_2}]}
        nodes = {
            'input_1': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
            'input_2': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
            'node_1': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'node_2': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'node_3': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'node_4': {'type': 'Identity', 'kind': 'op', 'op': 'NotPlaceholder'},
            'output': { 'kind': 'op', 'op': 'Result'},
            'input_3': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'}
        }
        edges = [
            ('input_1', 'node_1'),
            ('node_1', 'node_2'),
            ('node_3', 'output'),
            ('input_2', 'node_4'),
            ('node_4', 'output'),
            ('input_3', 'output')
        ]
        graph = build_graph(nodes, edges)
        self.assertRaises(Error, add_input_ops, graph, inputs, True)

    # Tests for cases with input/output ports cutting
    def test_add_input_with_input_port_before_infer(self):
        shape = np.array([1, 2, 3, 4])
        inputs = {'conv_1': [{'shape': shape, 'in': 0}]}
        nodes = {
            'old_input': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
            'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder'},
            'relu_1': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder'},
            'output': {'type': 'SoftMax', 'kind': 'op', 'op': 'NotPlaceholder'}
        }
        edges = [
            ('old_input', 'conv_1'),
            ('conv_1', 'relu_1'),
            ('relu_1', 'output')
        ]
        graph = build_graph(nodes, edges)
        add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=True)

        # Check that graph
        graph_ref = build_graph(nodes, edges, update_attributes={'old_input': {'shape': shape}})
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output')
        self.assertTrue(flag, resp)

        # also checks that new old_input was changed
        new_input = list(graph.in_edges('conv_1'))[0][0]
        self.assertFalse(graph.node['old_input']['is_input'])
        self.assertTrue(graph.node[new_input]['is_input'])
        self.assertTrue((new_input, 'conv_1') in graph.edges())
        self.assertTrue(('old_input', 'conv_1') not in graph.edges())

    def test_add_input_with_output_port_before_infer(self):
        shape = np.array([1, 2, 3, 4])
        inputs = {'conv_1': [{'shape': shape, 'out': 0}]}
        nodes = {
            'old_input': {'type': 'Identity', 'kind': 'op', 'op': 'Parameter'},
            'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder'},
            'conv_2': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder'},
            'relu_1': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder'},
            'output': {'type': 'SoftMax', 'kind': 'op', 'op': 'NotPlaceholder'}
        }
        edges = [
            ('old_input', 'conv_1'),
            ('conv_1', 'relu_1'),
            ('conv_2', 'relu_1'),
            ('relu_1', 'output')
        ]
        graph = build_graph(nodes, edges)
        add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=True)

        graph_ref = build_graph(nodes_attrs={'new_input': {'kind': 'op', 'op': 'Parameter', 'shape': shape},
                                             **nodes},
                                edges=[('new_input', 'relu_1'),
                                       ('relu_1', 'output'),
                                       ('conv_2', 'relu_1'),
                                       ('old_input', 'conv_1'),],)
        # Check that new input is added right (with right ports !)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='output')
        self.assertTrue(flag, resp)

        # Check that other graph is not damaged
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='conv_1')
        self.assertTrue(flag, resp)

        # Checks for new input and edges
        self.assertTrue('conv_1/placeholder_out_port_0' in graph.nodes())
        new_input = 'conv_1/placeholder_out_port_0'
        self.assertTrue(graph.node[new_input]['is_input'])
        self.assertTrue((new_input, 'relu_1') in graph.edges())
        self.assertTrue(('old_input', 'relu_1') not in graph.edges())

    def test_add_input_with_output_port_after_infer(self):
        shape = np.array([1, 2, 3, 4])
        inputs = {'conv_1': [{'shape': shape, 'out': 0}]}
        nodes = {
            'old_input': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'inp_data' : {'kind': 'data', 'shape': shape + 1},
            'conv_1': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder'},
            'conv_data': {'kind': 'data', 'shape': shape, 'value': None, 'data_attr': 'data_attr_value'},
            'relu_1': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder'},
        }
        edges = [
            ('old_input', 'inp_data'),
            ('inp_data', 'conv_1'),
            ('conv_1', 'conv_data'),
            ('conv_data', 'relu_1', {'edge_attr': 'edge_value'}),
        ]
        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        add_input_ops(graph=graph, user_defined_inputs=inputs, before_infer=False)

        graph_ref = build_graph(nodes_attrs={'new_input': {'kind': 'op', 'op': 'Parameter', 'shape': shape},
                                             **nodes},
                                edges=[('old_input', 'inp_data'),
                                       ('inp_data', 'conv_1'),
                                       ('new_input', 'conv_data'),
                                       ('conv_data', 'relu_1', {'edge_attr': 'edge_value'}),
                                       ],)
        # Check that new input is added right (with right ports !)
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='relu_1')
        self.assertTrue(flag, resp)

        # Check that other graph is not damaged
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='conv_1')
        self.assertTrue(flag, resp)

        # Checks for new input and edges
        self.assertTrue('conv_1/placeholder_out_port_0' in graph.nodes())
        new_input = 'conv_1/placeholder_out_port_0'

        self.assertTrue(graph.node[new_input]['is_input'])

        self.assertTrue(Node(graph, 'relu_1').in_node(0)['data_attr'] == 'data_attr_value')
        self.assertTrue(Node(graph, 'relu_1').in_edge(0)['edge_attr'] == 'edge_value')


class TestOutputCut():
    # {'embeddings': [{'port': None}]}
    @pytest.mark.parametrize("output",[{'C':[{'port': None}]}, {'C': [{'out': 0}]}, {'C': [{'out': 1}]}])
    def test_output_port_cut(self, output):
        nodes = {'A': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 'B': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 'C': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 'D': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 'E': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 }
        edges = [
            ('A', 'C', {'in': 0, 'out': 0}),
            ('B', 'C', {'in': 1, 'out': 0}),
            ('C', 'D', {'in': 0, 'out': 0}),
            ('C', 'E', {'in': 0, 'out': 1})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        sinks = add_output_ops(graph, output)
        graph.clean_up()
        assert len(Node(graph, 'C').out_nodes()) == 1
        assert len(Node(graph, 'C').in_nodes()) == 2

    @pytest.mark.parametrize("output",[{'C': [{'in': 0}]}, {'C': [{'in': 1}]}])
    def test_output_port_cut(self, output):
        nodes = {'A': {'op': 'Parameter', 'kind': 'op'},
                 'B': {'op': 'Parameter', 'kind': 'op'},
                 'C': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 'D': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 'E': {'type': 'Identity', 'kind': 'op', 'op': 'Identity'},
                 }
        edges = [
            ('A', 'C', {'in': 0, 'out': 0}),
            ('B', 'C', {'in': 1, 'out': 0}),
            ('C', 'D', {'in': 0, 'out': 0}),
            ('C', 'E', {'in': 0, 'out': 1})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        sinks = add_output_ops(graph, output)
        graph.clean_up()
        assert len(graph.nodes()) == 2


class TestUserDataRepack(UnitTestWithMockedTelemetry):
    nodes = {'A': {'name': 'Aa', 'op': 'Parameter', 'kind': 'op'},
             'B': {'name': 'Bb', 'op': 'Parameter', 'kind': 'op'},
             'C': {'name': 'Cc', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
             'D': {'name': 'Dd', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
             'E': {'name': 'Ee', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Identity'},
             }
    edges = [
        ('A', 'C', {'in': 0, 'out': 0}),
        ('B', 'C', {'in': 1, 'out': 0}),
        ('C', 'D', {'in': 0, 'out': 0}),
        ('C', 'E', {'in': 0, 'out': 1})
    ]

    def test_input_user_data_repack_none(self):
        graph = build_graph(self.nodes, self.edges)
        input, freeze_placeholder = input_user_data_repack(graph, None, None)
        self.assertEqual(input, None)
        self.assertEqual(freeze_placeholder, None)

    def test_input_user_data_repack_names_to_ids_list(self):
        graph = build_graph(self.nodes, self.edges)
        input, freeze_placeholder = input_user_data_repack(graph, ['Aa', 'Bb'], None)
        self.assertDictEqual(input, {'A': [{'shape': None, 'port': None}], 'B': [{'shape': None, 'port': None}]})
        self.assertEqual(freeze_placeholder, None)

    def test_input_user_data_repack_names_ports_in_out(self):
        graph = build_graph(self.nodes, self.edges)
        input, freeze_placeholder = input_user_data_repack(graph, ['Aa:0', '1:Cc'], None)
        self.assertDictEqual(input, {'A': [{'shape': None, 'out': 0}], 'C': [{'shape': None, 'in': 1}]})
        self.assertEqual(freeze_placeholder, None)

    def test_input_user_data_repack_dict_with_shapes(self):
        graph = build_graph(self.nodes, self.edges)
        shape_1 = np.array([1, 160, 160, 3])
        shape_2 = np.array([1, 127, 127, 3])
        input, freeze_placeholder = input_user_data_repack(graph, {'Aa': shape_1, 'Bb': shape_2}, None)
        self.assertDictEqual(input, {'A': [{'shape': shape_1, 'port': None}], 'B': [{'shape': shape_2, 'port': None}]})
        self.assertEqual(freeze_placeholder, None)

    def test_input_user_data_repack_dict_with_shapes_and_ports(self):
        graph = build_graph(self.nodes, self.edges)
        shape_1 = np.array([1, 160, 160, 3])
        shape_2 = np.array([1, 127, 127, 3])
        input, freeze_placeholder = input_user_data_repack(graph, {'Aa:0': shape_1, 'Bb:0': shape_2}, None)
        self.assertDictEqual(input, {'A': [{'shape': shape_1, 'out': 0}], 'B': [{'shape': shape_2, 'out': 0}]})
        self.assertEqual(freeze_placeholder, None)

    def test_freeze_placeholder_and_input(self):
        graph = build_graph(self.nodes, self.edges)
        shape_1 = np.array([1, 160, 160, 3])
        input, freeze_placeholder = input_user_data_repack(graph, {'Aa:0': shape_1}, {'Bb': False})
        self.assertDictEqual(input, {'A': [{'shape': shape_1, 'out': 0}], 'B': [{'shape': None, 'port': None}]})
        self.assertEqual(freeze_placeholder, {'B': False})

    def test_error(self):
        graph = build_graph(self.nodes, self.edges)
        self.assertRaises(Error, input_user_data_repack, graph, PartialShape([1, 227, 227, 3]), None)

    def test_error_2(self):
        graph = build_graph(self.nodes, self.edges)
        self.assertRaises(Error, input_user_data_repack, graph, PartialShape([1, 227, 227, 3]), None)

    def test_error_3(self):
        graph = build_graph(self.nodes, self.edges)
        self.assertRaises(Error, input_user_data_repack, graph, ['Bcb'], None)

    def test_input_and_freeze(self):
        graph = build_graph(self.nodes, self.edges)
        shape_1 = PartialShape([1, 160, 160, 3])
        input, freeze_placeholder = input_user_data_repack(graph, shape_1, {'Bb': True})
        self.assertDictEqual(input, {'A': [{'shape': shape_1, 'port': None}], 'B': [{'shape': None, 'port': None}]})
        self.assertDictEqual(freeze_placeholder, {'B': True})

    def test_freeze_new_placeholder_1(self):
        # create a new placeholder Cc:0 by cutting output port with shape_2 = [5] and freeze a value [1.0 1.0 2.0 3.0 5.0]
        graph = build_graph(self.nodes, self.edges)
        shape_1 = np.array([1, 160, 160, 3])
        shape_2 = np.array([5])
        input, freeze_placeholder = input_user_data_repack(graph, {'Aa:0': shape_1, 'Cc:0' : shape_2}, {'Bb': False, 'Cc:0' : [1.0, 1.0, 2.0, 3.0, 5.0]})
        self.assertDictEqual(input, {'A' : [{'shape' : shape_1, 'out' : 0}], 'B' : [{'shape' : None, 'port' : None}], 'C' : [{'shape' : shape_2, 'out' : 0}]})
        self.assertEqual(freeze_placeholder, {'B' : False, 'C/placeholder_out_port_0' : [1.0, 1.0, 2.0, 3.0, 5.0]})

    def test_freeze_new_placeholder_2(self):
        # create a new placeholder Ee by cutting input port with shape_2 = [2, 2] and freeze a value [[1.0, 1.0], [2.0, 3.0]]
        graph = build_graph(self.nodes, self.edges)
        shape_1 = np.array([1, 160, 160, 3])
        shape_2 = np.array([2, 2])
        input, freeze_placeholder = input_user_data_repack(graph, {'Aa:0': shape_1, 'Ee' : shape_2}, {'Bb': False, 'Ee' : [[1.0, 1.0], [2.0, 3.0]]})
        self.assertDictEqual(input, {'A' : [{'shape' : shape_1, 'out' : 0}], 'B' : [{'shape' : None, 'port' : None}], 'E' : [{'shape' : shape_2, 'port' : None}]})
        self.assertEqual(freeze_placeholder, {'B' : False, 'E/placeholder_port_None' : [[1.0, 1.0], [2.0, 3.0]]})

    def test_freeze_new_placeholder_error(self):
        # shape is not specified for new placeholder Cc:0 with frozen value
        graph = build_graph(self.nodes, self.edges)
        shape_1 = np.array([1, 160, 160, 3])
        self.assertRaises(Error, input_user_data_repack, graph, {'Aa:0': shape_1}, {'Bb': False, 'Cc:0' : [1.0, 1.0, 2.0, 3.0, 5.0]})

    def test_output_user_data_repack(self):
        graph = build_graph(self.nodes, self.edges)
        output = output_user_data_repack(graph, ['Cc'])
        self.assertDictEqual(output, {'C': [{'port': None}]})

    def test_output_user_data_repack_ports(self):
        graph = build_graph(self.nodes, self.edges)
        output = output_user_data_repack(graph, ['Cc:1', '0:Cc'])
        self.assertDictEqual(output, {'C': [{'out': 1}, {'in': 0}]})

    def test_output_user_data_repack_none(self):
        graph = build_graph(self.nodes, self.edges)
        output = output_user_data_repack(graph, None)
        self.assertEqual(output, None)


class TestExtractPort(unittest.TestCase):
    def setUp(self) -> None:
        nodes = {
            'input_id': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter', 'name': '1input1:0'},
            'conv_id': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder', 'name': '1input1'},
            'relu_id': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder', 'name': 'relu'},
            'squeeze_id': {'type': 'Squeeze', 'kind': 'op', 'op': 'NotPlaceholder', 'name': 'relu:0'},
        }
        edges = [
            ('input_id', 'conv_id'),
            ('conv_id', 'relu_id'),
            ('relu_id', 'squeeze_id'),
        ]
        self.graph = build_graph(nodes, edges)

    def test_out_port(self):
        node_id, direction, port = get_node_id_with_ports(self.graph, '1input1:0:0')
        self.assertEqual(node_id, 'input_id')
        self.assertEqual(direction, 'out')
        self.assertEqual(port, 0)

    def test_in_port1(self):
        node_id, direction, port = get_node_id_with_ports(self.graph, '0:1input1')
        self.assertEqual(node_id, 'conv_id')
        self.assertEqual(direction, 'in')
        self.assertEqual(port, 0)

    def test_in_port2(self):
        node_id, direction, port = get_node_id_with_ports(self.graph, '0:relu:0')
        self.assertEqual(node_id, 'squeeze_id')
        self.assertEqual(direction, 'in')
        self.assertEqual(port, 0)

    def test_no_port1(self):
        node_id, direction, port = get_node_id_with_ports(self.graph, '1input1')
        self.assertEqual(node_id, 'conv_id')
        self.assertEqual(direction, 'port')
        self.assertEqual(port, None)

    def test_no_port2(self):
        self.assertRaises(Error, get_node_id_with_ports, self.graph, '1input1:0')

    def test_non_int(self):
        self.assertRaises(Error, get_node_id_with_ports, self.graph, 'port:1input1')

    def test_two_ports(self):
        self.assertRaises(Error, get_node_id_with_ports, self.graph, '0:1input1:1')

    def test_name_looks_like_port_number(self):
        nodes = {
            'input_id': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter', 'name': '0'},
            'conv_id': {'type': 'Convolution', 'kind': 'op', 'op': 'NotPlaceholder', 'name': '1'},
            'relu_id': {'type': 'ReLU', 'kind': 'op', 'op': 'NotPlaceholder', 'name': '2'},
        }
        edges = [
            ('input_id', 'conv_id'),
            ('conv_id', 'relu_id'),
        ]
        graph = build_graph(nodes, edges)
        node_id, direction, port = get_node_id_with_ports(graph, '0:2')
        self.assertEqual(node_id, 'relu_id')
        self.assertEqual(direction, 'in')
        self.assertEqual(port, 0)


class TestCaffePythonFrontExtractorOp(unittest.TestCase):
    def test_get_attrs(self):
        exp_attrs = {"test_attr_1": 12, "test_attr_2": "sdf sdf"}
        param_str = "'test_attr_1': 12, 'test_attr_2': 'sdf sdf'"
        attrs = CaffePythonFrontExtractorOp.get_attrs(FakePythonParam(FakeMultiParam({'param_str': param_str})))
        self.assertEqual(exp_attrs, attrs)

class TestBoolToSrtFunction(unittest.TestCase):
    def test_bool_to_str(self):
        graph = build_graph(nodes_attributes,
                            [('input', 'pool_1'),
                             ('pool_1', 'output'),
                             ('output', 'op_output')
                             ],
                            {'pool_1': {'bool_attr': None}
                             })
        pool_1_node = Node(graph, 'pool_1')
        attrs = [(True, 'true'), (False, 'false'), (1, 'true'), (0, 'false')]
        for attr in attrs:
            pool_1_node.bool_attr = attr[0]
            self.assertEqual(attr[1], bool_to_str(pool_1_node, 'bool_attr'))
