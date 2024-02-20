# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, \
    dynamic_dimension_value, dynamic_dimension, strict_compare_tensors, mo_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.split import AttributedSplit, AttributedVariadicSplit, VariadicSplit
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class TestSplitOp(unittest.TestCase):
    nodes = {
        'input': {'kind': 'op'},
        'split_input_data': {'kind': 'data', 'shape': None, 'value': None},
        'split_op': {'kind': 'op', 'axis': None, 'num_splits': None, 'op': 'AttributedSplit'},
        'split_output_0_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_0': {'kind': 'op'},
        'split_output_1_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_1': {'kind': 'op'},
    }
    edges = [
        ('input', 'split_input_data'),
        ('split_input_data', 'split_op'),
        ('split_op', 'split_output_0_data'),
        ('split_output_0_data', 'output_0'),
        ('split_op', 'split_output_1_data'),
        ('split_output_1_data', 'output_1'),
    ]

    def test_split_shape_infer(self):
        #  test configuration
        input_shape = [2, 10]
        input_value = None
        axis = 1
        num_splits = 2
        output_shape = [2, 5]
        output_value = [None, None]

        # action
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array(input_shape),
                                                     'value': input_value},
                                'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                            }
                            )

        split_op = Node(graph, 'split_op')
        AttributedSplit.infer(split_op)

        # reference
        graph_ref = build_graph(self.nodes, self.edges,
                                {
                                    'split_input_data': {'shape': int64_array(input_shape),
                                                         'value': input_value},
                                    'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                                    'split_output_0_data': {'shape': int64_array(output_shape),
                                                            'value': output_value[0]},
                                    'split_output_1_data': {'shape': int64_array(output_shape),
                                                            'value': output_value[1]},
                                }
                                )

        # check
        (flag, resp) = compare_graphs(graph, graph_ref, 'split_input_data')
        self.assertTrue(flag, resp)

    def test_split_dynamic_shape_infer(self):
        #  test configuration
        input_shape = [2, dynamic_dimension_value]
        input_value = None
        axis = 1
        num_splits = 2
        output_shape = [2, dynamic_dimension_value]
        output_value = [None, None]

        # action
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': shape_array(input_shape),
                                                     'value': input_value},
                                'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                            }
                            )

        split_op = Node(graph, 'split_op')
        AttributedSplit.infer(split_op)

        # reference
        graph_ref = build_graph(self.nodes, self.edges,
                                {
                                    'split_input_data': {'shape': shape_array(input_shape),
                                                         'value': input_value},
                                    'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                                    'split_output_0_data': {'shape': shape_array(output_shape),
                                                            'value': output_value[0]},
                                    'split_output_1_data': {'shape': shape_array(output_shape),
                                                            'value': output_value[1]},
                                }
                                )

        # check
        (flag, resp) = compare_graphs(graph, graph_ref, 'split_input_data')
        self.assertTrue(flag, resp)
        self.assertTrue(strict_compare_tensors(Node(graph, 'split_output_0_data').shape, shape_array(output_shape)))

    def test_split_value_infer(self):
        #  test configuration
        input_shape = [2, 10]
        input_value = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
        axis = 1
        num_splits = 2
        output_shape = [2, 5]
        output_value = [[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14]], [[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]]]

        # action
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array(input_shape),
                                                     'value': int64_array(input_value)},
                                'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                            }
                            )

        split_op = Node(graph, 'split_op')
        AttributedSplit.infer(split_op)

        # reference
        graph_ref = build_graph(self.nodes, self.edges,
                                {
                                    'split_input_data': {'shape': int64_array(input_shape),
                                                         'value': int64_array(input_value)},
                                    'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                                    'split_output_0_data': {'shape': int64_array(output_shape),
                                                            'value': int64_array(output_value[0])},
                                    'split_output_1_data': {'shape': int64_array(output_shape),
                                                            'value': int64_array(output_value[1])},
                                }
                                )

        # check
        (flag, resp) = compare_graphs(graph, graph_ref, 'split_input_data')
        self.assertTrue(flag, resp)


class TestAttributedVariadicSplitOp(unittest.TestCase):
    nodes = {
        'input': {'kind': 'op'},
        'split_input_data': {'kind': 'data', 'shape': None, 'value': None},
        'split_op': {'kind': 'op', 'axis': None, 'split_lengths': None, 'op': 'AttributedVariadicSplit'},
        'split_output_0_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_0': {'kind': 'op'},
        'split_output_1_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_1': {'kind': 'op'},
        'split_output_2_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_2': {'kind': 'op'},
    }
    edges = [
        ('input', 'split_input_data'),
        ('split_input_data', 'split_op'),
        ('split_op', 'split_output_0_data'),
        ('split_output_0_data', 'output_0'),
        ('split_op', 'split_output_1_data'),
        ('split_output_1_data', 'output_1'),
        ('split_op', 'split_output_2_data'),
        ('split_output_2_data', 'output_2'),
    ]

    def test_splitv_zero(self):
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([2, 13, 10, 0]),
                                             'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        AttributedVariadicSplit.infer(node)

        self.assertTrue(len(node.out_edges()) == 3)
        self.assertTrue(np.all(node.split_lengths == np.array([2, 13, 10])))

    def test_splitv_dynamic_input(self):
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': shape_array([2, 12, dynamic_dimension_value, 30])},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([2, 13, 10]),
                                             'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        AttributedVariadicSplit.infer(node)

        self.assertTrue(len(node.out_edges()) == 3)
        self.assertTrue(np.all(node.split_lengths == np.array([2, 13, 10])))

    def test_splitv_zero_not_last(self):
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([2, 13, 0, 10]),
                                             'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')

        # extractor should do it
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)
        node.out_port(2).get_connection().set_source(node.out_port(3))

        AttributedVariadicSplit.infer(node)

        self.assertTrue(node.out_port(3).disconnected())
        self.assertTrue(np.all(node.split_lengths == np.array([2, 13, 10])))

    def test_splitv_2_zero_not_last(self):
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([2, 13, 0, 0, 10]),
                                             'out_ports_count': 5},
                            }
                            )
        node = Node(graph, 'split_op')

        # extractor should do it
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)
        node.out_port(2).get_connection().set_source(node.out_port(4))

        AttributedVariadicSplit.infer(node)

        self.assertTrue(node.out_port(4).disconnected())
        self.assertTrue(node.out_port(3).disconnected())
        self.assertTrue(np.all(node.split_lengths == np.array([2, 13, 10])))


class TestVariadicSplitOp():
    nodes = {
        'input': {'kind': 'op'},
        'split_input_data': {'kind': 'data', 'shape': None, 'value': None},
        'split_axis': {'kind': 'op', 'op': 'Const'},
        'split_axis_data': {'kind': 'data', 'shape': None, 'value': None},
        'split_lengths': {'kind': 'op', 'op': 'Const'},
        'split_lengths_data': {'kind': 'data', 'shape': None, 'value': None},
        'split_op': {'kind': 'op', 'op': 'VariadicSplit'},
        'split_output_0_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_0': {'kind': 'op'},
        'split_output_1_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_1': {'kind': 'op'},
        'split_output_2_data': {'kind': 'data', 'shape': None, 'value': None},
        'output_2': {'kind': 'op'},
    }
    edges = [
        ('input', 'split_input_data'),
        ('split_input_data', 'split_op'),
        ('split_axis', 'split_axis_data'),
        ('split_axis_data', 'split_op'),
        ('split_lengths', 'split_lengths_data'),
        ('split_lengths_data', 'split_op'),
        ('split_op', 'split_output_0_data'),
        ('split_output_0_data', 'output_0'),
        ('split_op', 'split_output_1_data'),
        ('split_output_1_data', 'output_1'),
        ('split_op', 'split_output_2_data'),
        ('split_output_2_data', 'output_2'),
    ]

    @pytest.mark.parametrize("axis",[int64_array(2),
                int64_array([2])])
    def test_variadic_split_axis(self, axis):
        lengths = int64_array([2, 13, 10])
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_axis_data': {'value': axis},
                                'split_lengths_data': {'value': lengths},
                                'split_op': {'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        VariadicSplit.infer(node)

        ont_nodes_count = len(node.out_edges())
        assert ont_nodes_count == 3
        for out in range(ont_nodes_count):
            assert np.all(node.out_node(out).shape == int64_array([2, 12, lengths[out], 30]))

    def test_variadic_split_value_inference_with_uint32(self):
        axis = int64_array(2)
        # because sum of Python int and Numpy np.uint64 gives float64
        
        # but np.split accepts only integers and raises error for floats
        # therefore needed to explicitly cast np.split arguments into integer
        # added this test for that case
        lengths = mo_array([2, 13, 10], dtype=np.uint64)
        input_shape = mo_array([2, 12, 25, 30])
        input_value = np.zeros(input_shape)

        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': input_shape, 'value': input_value},
                                'split_axis_data': {'value': axis},
                                'split_lengths_data': {'value': lengths},
                                'split_op': {'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        VariadicSplit.infer(node)

        ont_nodes_count = len(node.out_edges())
        assert ont_nodes_count == 3
        for out in range(ont_nodes_count):
            assert np.all(node.out_node(out).shape == int64_array([2, 12, lengths[out], 30]))

    @pytest.mark.parametrize("axis",[int64_array([[2], [2]]),
                int64_array([2, 2])])
    def test_negative_variadic_split_axis(self, axis):
        lengths = int64_array([2, 13, 10])
        graph = build_graph(self.nodes, self.edges,
                            {
                                'split_input_data': {'shape': int64_array([2, 12, 25, 30])},
                                'split_axis_data': {'value': axis},
                                'split_lengths_data': {'value': lengths},
                                'split_op': {'out_ports_count': 4},
                            }
                            )
        node = Node(graph, 'split_op')
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        try:
            VariadicSplit.infer(node)
        except AssertionError as e:
            assert e.args[0] == 'VariadicSplit `axis` should be scalar or tensor with shape [1], '\
                                         'but it`s not for node split_op'


class TestSplitReverseInfer(unittest.TestCase):

    def test_split_reverse_infer(self):
        ref_input_shape = [7, 4, 6]
        axis = 2
        num_splits = 2
        output_shape_1 = [dynamic_dimension, 4, 3]
        output_shape_2 = [7, dynamic_dimension, 3]

        graph = build_graph(TestSplitOp.nodes, TestSplitOp.edges,
                            {
                                'split_input_data': {'shape': None,
                                                     'value': None},
                                'split_op': {'axis': np.array(axis), 'num_splits': np.array(num_splits)},
                                'split_output_0_data': {'shape': shape_array(output_shape_1),
                                                        'value': None},
                                'split_output_1_data': {'shape': shape_array(output_shape_2),
                                                        'value': None},
                            })

        split_node = Node(graph, 'split_op')
        AttributedSplit.reverse_infer(split_node)
        actual_input_shape = split_node.in_port(0).data.get_shape()
        self.assertTrue(strict_compare_tensors(ref_input_shape, actual_input_shape))


class TestAttributedVariadicSplitReverseInfer(unittest.TestCase):

    def test_splitv_dynamic_input(self):
        ref_input_shape = [7, 4, 11]
        axis = 2
        num_splits = 2
        output_shape_1 = [dynamic_dimension, 4, 3]
        output_shape_2 = [7, dynamic_dimension, 3]
        output_shape_3 = [7, dynamic_dimension, 5]

        graph = build_graph(TestAttributedVariadicSplitOp.nodes, TestAttributedVariadicSplitOp.edges,
                            {
                                'split_input_data': {'shape': None},
                                'split_op': {'axis': np.array(2), 'split_lengths': np.array([3, 3, 5]),
                                             'out_ports_count': 2},
                                'split_output_0_data': {'shape': shape_array(output_shape_1),
                                                        'value': None},
                                'split_output_1_data': {'shape': shape_array(output_shape_2),
                                                        'value': None},
                                'split_output_2_data': {'shape': shape_array(output_shape_3),
                                                        'value': None},
                            }
                            )
        node = Node(graph, 'split_op')
        for p in range(len(node.out_edges()), node.out_ports_count):
            node.add_output_port(p)

        AttributedVariadicSplit.reverse_infer(node)

        actual_input_shape = node.in_port(0).data.get_shape()
        self.assertTrue(strict_compare_tensors(ref_input_shape, actual_input_shape))
