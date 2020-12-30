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
from argparse import Namespace

import numpy as np

from extensions.middle.AddMeanScaleValues import AddMeanScaleValues
from extensions.middle.ScaleInput import ScaleInput
from mo.graph.graph import Graph, Node
from mo.utils.cli_parser import get_mean_scale_dictionary, parse_tuple_pairs
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_shaped_data, result, connect, connect_data, \
    valued_const_with_data

nodes = {
    **regular_op_with_shaped_data('parameter', [1, 3, 227, 227],
                                  {'type': 'Parameter', 'op': 'Parameter', 'shape': [1, 3, 227, 227]}),
    **regular_op_with_shaped_data('parameter_2', [1, 3, 227, 227],
                                  {'type': 'Parameter', 'op': 'Parameter', 'shape': [1, 3, 227, 227]}),

    **regular_op_with_shaped_data('mul_scale', [1, 3, 227, 227], {'type': 'Multiply', 'op': 'Mul'}),
    **regular_op_with_shaped_data('add_mean', [1, 3, 227, 227], {'type': 'Add', 'op': 'Add'}),

    **valued_const_with_data('scale', np.array([1. / 1., 1. / 2., 1. / 3.]).reshape((1, 3, 1, 1))),
    **valued_const_with_data('mean', np.array([-1., -2., -3.]).reshape((1, 3, 1, 1))),

    **regular_op_with_shaped_data('shape_of', [4], {'type': 'ShapeOf', 'op': 'ShapeOf'}),
    **regular_op_with_shaped_data('op', [1, 3, 227, 227], {}),
    **result('result'),
    **result('result_2'),
}


class AddMeanScaleValuesTest(unittest.TestCase):
    def check_graph_attrs(self, graph: Graph, graph_ref: Graph, parameter_node_names: list):
        for node in graph.get_op_nodes():
            if node.soft_get('name') in parameter_node_names:
                self.assertTrue(node.soft_get('type') == 'Parameter')
                out_node = node.out_node(0)
                out_node_ref = Node(graph_ref, node.id).out_node(0)
                self.assertTrue(out_node['fw_tensor_debug_info'] == out_node_ref['fw_tensor_debug_info'])
            else:
                if 0 in node.out_nodes():
                    out_node = node.out_node(0)
                    self.assertFalse('fw_tensor_debug_info' in out_node)

    def set_graph_attrs(self, graph: Graph, parameter_node_names: list):
        for node in graph.get_op_nodes():
            if node.soft_get('name') in parameter_node_names:
                self.assertTrue(node.soft_get('type') == 'Parameter')
                out_node = node.out_node(0)
                out_node['fw_tensor_debug_info'] = ['fw_name', 0]

    def test_mean_values_with_data_name(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:add_mean'),
            *connect('mean', '1:add_mean'),
            *connect('add_mean', 'result'),
        ])

        mean_values = parse_tuple_pairs('(1,2,3)')
        scale_values = parse_tuple_pairs('')
        mean_scale = get_mean_scale_dictionary(mean_values, scale_values, None)
        argv = Namespace(mean_scale_values=mean_scale)

        graph = build_graph(nodes, [*connect('parameter', 'result')], nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['parameter'])
        self.set_graph_attrs(graph_ref, ['parameter'])
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter'])

    def test_mean_values_without_data_name(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:add_mean'),
            *connect('mean', '1:add_mean'),
            *connect('add_mean', 'result'),
        ], {'parameter': {'name': 'None'}})

        mean_values = parse_tuple_pairs('(1,2,3)')
        scale_values = parse_tuple_pairs('')
        mean_scale = get_mean_scale_dictionary(mean_values, scale_values, None)
        argv = Namespace(mean_scale_values=mean_scale)

        graph = build_graph(nodes, [*connect('parameter', 'result')], {'parameter': {'name': 'None'}},
                            nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['None'])
        self.set_graph_attrs(graph_ref, ['None'])
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['None'])

    def test_mean_values_explicit_and_optimized(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:add_mean'),
            *connect('mean', '1:add_mean'),
            *connect('add_mean', 'result'),
            *connect('parameter_2', 'result_2'),
        ])

        argv = Namespace(mean_scale_values={'parameter': {'mean': np.array([1., 2., 3.])},
                                            'parameter_2': {'mean': np.array([0., 0., 0.])}})
        graph = build_graph(nodes, [*connect('parameter', 'result'), *connect('parameter_2', 'result_2')],
                            nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['parameter', 'parameter_2'])
        self.set_graph_attrs(graph_ref, ['parameter', 'parameter_2'])
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result_2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter', 'parameter_2'])

    def test_mean_values_explicit_and_scale_values_optimized(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:add_mean'),
            *connect('mean', '1:add_mean'),
            *connect('add_mean', 'result'),
        ])

        argv = Namespace(mean_scale_values={'parameter': {'scale': np.array([1.]), 'mean': np.array([1., 2., 3.])}})
        graph = build_graph(nodes, [*connect('parameter', 'result')], nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['parameter'])
        self.set_graph_attrs(graph_ref, ['parameter'])
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter'])

    def test_mean_values_optimized_and_scale_values_explicit(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:mul_scale'),
            *connect('scale', '1:mul_scale'),
            *connect('mul_scale', 'result'),
        ])

        argv = Namespace(
            mean_scale_values={'parameter': {'scale': np.array([1., 2., 3.]), 'mean': np.array([0., 0., 0.])}})
        graph = build_graph(nodes, [*connect('parameter', 'result')], nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['parameter'])
        self.set_graph_attrs(graph_ref, ['parameter'])
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter'])

    def test_mean_values_explicit_and_scale_values_explicit(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:add_mean'),
            *connect('mean', '1:add_mean'),
            *connect('add_mean', '0:mul_scale'),
            *connect('scale', '1:mul_scale'),
            *connect('mul_scale', 'result'),
        ])

        argv = Namespace(mean_scale_values=[[np.array([1., 2., 3.]), np.array([1., 2., 3.])]])
        graph = build_graph(nodes, [*connect('parameter', 'result')],
                            nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['parameter'])
        self.set_graph_attrs(graph_ref, ['parameter'])
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter'])

    def test_mean_values_explicit_and_scale_values_explicit_on_cutted_graph(self):
        """
        Test case when user cutted start of the network and specified mean/scale value to the new input node 'node_3'.
        """
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:add_mean'),
            *connect('mean', '1:add_mean'),
            *connect('add_mean', 'result'),

            *connect('parameter_2', '0:mul_scale'),
            *connect('scale', '1:mul_scale'),
            *connect('mul_scale', 'op'),
            *connect('op', 'result_2'),
        ])

        argv = Namespace(
            mean_scale_values={'parameter': {'mean': np.array([1, 2, 3])}, 'op': {'scale': np.array([1, 2, 3])}})
        graph = build_graph(
            nodes, [*connect('parameter', 'result'), *connect('parameter_2', 'op'), *connect('op', 'result_2')],
            {'parameter_2': {'initial_node_name': 'op'}}, nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['parameter', 'parameter_2'])
        self.set_graph_attrs(graph_ref, ['parameter', 'parameter_2'])
        graph.graph['layout'] = 'NCHW'
        AddMeanScaleValues().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result_2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter', 'parameter_2'])

    def test_mean_values_explicit_and_scale_values_explicit_with_shape_of(self):
        graph_ref = build_graph(nodes,
                                [
                                    *connect('parameter', '0:add_mean'),
                                    *connect('mean', '1:add_mean'),
                                    *connect('add_mean', '0:mul_scale'),
                                    *connect('scale', '1:mul_scale'),
                                    *connect('mul_scale', 'result'),
                                    *connect_data('parameter', 'shape_of'),
                                    *connect('shape_of', 'result_2'),
                                ],
                                nodes_with_edges_only=True)

        argv = Namespace(
            mean_scale_values={'parameter': {'mean': np.array([1, 2, 3]), 'scale': np.array([1, 2, 3])}})
        graph = build_graph(nodes,
                            [
                                *connect('parameter', 'result'),
                                *connect_data('parameter', 'shape_of'),
                                *connect('shape_of', 'result_2'),
                            ],
                            nodes_with_edges_only=True, cli=argv)
        self.set_graph_attrs(graph, ['parameter'])
        self.set_graph_attrs(graph_ref, ['parameter'])
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result_2', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter'])

    def test_scale_input(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:mul_scale'),
            *connect('scale', '1:mul_scale'),
            *connect('mul_scale', 'result'),
        ], {'scale': {'shape': [1, 1, 1, 1], 'value': np.array(1 / 255)},
            'scale_d': {'shape': [1, 1, 1, 1], 'value': np.array(1 / 255)}})

        graph = build_graph(nodes, connect('parameter', 'result'), nodes_with_edges_only=True, cli=Namespace(scale=255))
        self.set_graph_attrs(graph, ['parameter'])
        self.set_graph_attrs(graph_ref, ['parameter'])
        graph.graph['layout'] = 'NCHW'

        ScaleInput().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter'])

    def test_scale_input_2(self):
        graph_ref = build_graph(nodes, connect('parameter', 'result'), nodes_with_edges_only=True)
        graph = build_graph(nodes, connect('parameter', 'result'), nodes_with_edges_only=True, cli=Namespace(scale=1))
        self.set_graph_attrs(graph, ['parameter'])
        self.set_graph_attrs(graph_ref, ['parameter'])
        graph.graph['layout'] = 'NCHW'

        ScaleInput().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, ['parameter'])

    def test_debug_info_absence(self):
        graph_ref = build_graph(nodes, [
            *connect('parameter', '0:add_mean'),
            *connect('mean', '1:add_mean'),
            *connect('add_mean', '0:mul_scale'),
            *connect('scale', '1:mul_scale'),
            *connect('mul_scale', 'result'),
        ])

        argv = Namespace(mean_scale_values=[[np.array([1., 2., 3.]), np.array([1., 2., 3.])]])
        graph = build_graph(nodes, [*connect('parameter', 'result')],
                            nodes_with_edges_only=True, cli=argv)
        graph.graph['layout'] = 'NCHW'

        AddMeanScaleValues().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.check_graph_attrs(graph, graph_ref, [])
