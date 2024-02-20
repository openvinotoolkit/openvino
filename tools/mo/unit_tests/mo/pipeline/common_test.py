# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
import pytest
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.pipeline.common import determined_sort, get_fw_tensor_debug_info, get_sorted_outputs
from unit_tests.utils.graph import build_graph_with_edge_attrs


class TestTopologicalSort():
    @pytest.mark.parametrize( "edges",[
        [('A', 'Ad', {'out': 0}),
         ('Ad', 'B', {'in': 0}),
         ('B', 'Bd', {'out': 0}),
         ('Bd', 'C', {'in': 0}),
         ('C', 'Cd', {'out': 0}),
         ('Cd', 'D', {'in': 0}),
         ('D', 'Dd', {'out': 0}),
         ('Dd', 'E', {'in': 0}),
         ('E', 'Ed', {'out': 0}),
         ('Ed', 'I', {'in': 0}),
         ('Cd', 'F', {'in': 0}),
         ('F', 'Fd', {'out': 0}),
         ('Fd', 'G', {'in': 0}),
         ('G', 'Gd', {'out': 0}),
         ('Gd', 'I', {'in': 1}),
         ('Cd', 'H', {'in': 0}),
         ('H', 'Hd', {'out': 0}),
         ('Hd', 'I', {'in': 2}),
         ('I', 'Id', {'out': 0}),
         ('Id', 'J', {'in': 0}),
         ('J', 'Jd', {'out': 0}),
         ('Jd', 'K', {'in': 0}),
         ('K', 'Kd', {'out': 0})],

        [('A', 'Ad', {'out': 0}),
         ('Ad', 'B', {'in': 0}),
         ('B', 'Bd', {'out': 0}),
         ('Bd', 'C', {'in': 0}),
         ('C', 'Cd', {'out': 0}),
         ('Cd', 'D', {'in': 0}),
         ('D', 'Dd', {'out': 0}),
         ('Dd', 'E', {'in': 0}),
         ('E', 'Ed', {'out': 0}),
         ('Ed', 'I', {'in': 0}),
         ('Cd', 'F', {'in': 0}),
         ('F', 'Fd', {'out': 0}),
         ('Fd', 'G', {'in': 0}),
         ('G', 'Gd', {'out': 0}),
         ('Gd', 'I', {'in': 1}),
         ('Cd', 'H', {'in': 0}),
         ('H', 'Hd', {'out': 0}),
         ('Hd', 'I', {'in': 2}),
         ('I', 'Id', {'out': 0}),
         ('Id', 'J', {'in': 0}),
         ('J', 'Jd', {'out': 0}),
         ('Jd', 'K', {'in': 0}),
         ('K', 'Kd', {'out': 0}),
         ('Ad', 'E', {'in': 1}),
         ('Bd', 'K', {'in': 1}),
         ('Hd', 'J', {'in': 1})],

        [('A', 'Ad', {'out': 0}),
         ('Ad', 'B', {'in': 0}),
         ('B', 'Bd', {'out': 0}),
         ('Bd', 'C', {'in': 0}),
         ('C', 'Cd', {'out': 0}),
         ('Cd', 'D', {'in': 0}),
         ('D', 'Dd', {'out': 0}),
         ('Dd', 'E', {'in': 0}),
         ('E', 'Ed', {'out': 0}),
         ('Ed', 'I', {'in': 0}),
         ('Cd', 'F', {'in': 0}),
         ('F', 'Fd', {'out': 0}),
         ('Fd', 'G', {'in': 0}),
         ('G', 'Gd', {'out': 0}),
         ('Gd', 'I', {'in': 1}),
         ('Cd', 'H', {'in': 0}),
         ('H', 'Hd', {'out': 0}),
         ('Hd', 'I', {'in': 2}),
         ('I', 'Id', {'out': 0}),
         ('Id', 'J', {'in': 0}),
         ('J', 'Jd', {'out': 0}),
         ('Jd', 'K', {'in': 0}),
         ('K', 'Kd', {'out': 0}),
         ('Ad', 'E', {'in': 1}),
         ('Bd', 'K', {'in': 1}),
         ('Hd', 'J', {'in': 1}),
         ('Dd', 'F', {'in': 1}),
         ('Fd', 'H', {'in': 1}),
         ('Gd', 'H', {'in': 0})]]
    )
    def test_determined_topological_sort(self, edges):
        nodes = {'A': {'type': 'Identity', 'kind': 'op'},
                 'B': {'type': 'Identity', 'kind': 'op'},
                 'C': {'type': 'Identity', 'kind': 'op'},
                 'D': {'type': 'Identity', 'kind': 'op'},
                 'E': {'type': 'Identity', 'kind': 'op'},
                 'F': {'type': 'Identity', 'kind': 'op'},
                 'G': {'type': 'Identity', 'kind': 'op'},
                 'H': {'type': 'Identity', 'kind': 'op'},
                 'I': {'type': 'Identity', 'kind': 'op'},
                 'J': {'type': 'Identity', 'kind': 'op'},
                 'K': {'type': 'Identity', 'kind': 'op'},
                 'Ad': {'value': None, 'kind': 'data'},
                 'Bd': {'value': None, 'kind': 'data'},
                 'Cd': {'value': None, 'kind': 'data'},
                 'Dd': {'value': None, 'kind': 'data'},
                 'Ed': {'value': None, 'kind': 'data'},
                 'Fd': {'value': None, 'kind': 'data'},
                 'Gd': {'value': None, 'kind': 'data'},
                 'Hd': {'value': None, 'kind': 'data'},
                 'Id': {'value': None, 'kind': 'data'},
                 'Jd': {'value': None, 'kind': 'data'},
                 'Kd': {'value': None, 'kind': 'data'},
                 }

        graph = build_graph_with_edge_attrs(nodes, edges)
        outputs = [Node(graph, 'Kd')]
        for i in range(100):
            op_order, data_order = determined_sort(outputs)
            assert op_order == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            assert data_order == ['Ad', 'Bd', 'Cd', 'Dd', 'Ed', 'Fd', 'Gd', 'Hd', 'Id', 'Jd', 'Kd']


class TestGetFWTensorName(unittest.TestCase):
    def test_get_fw_tensor_debug_info(self):
        nodes = {
            'A': {'type': 'Identity', 'kind': 'op'},
            'B': {'type': 'Identity', 'kind': 'op'},
            'C': {'type': 'Identity', 'kind': 'op'},
            'Ad': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('A', 0)]},
            'Bd': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('B', 0)]},
            'Cd': {'value': None, 'kind': 'data'},
        }
        edges = [
            ('A', 'Ad', {'out': 0}),
            ('Ad', 'B', {'in': 0}),
            ('B', 'Bd', {'out': 0}),
            ('Bd', 'C', {'in': 0}),
            ('C', 'Cd', {'out': 0})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        fw_debug_info = get_fw_tensor_debug_info(Node(graph, 'Cd'))
        self.assertEqual(len(fw_debug_info), 1)
        self.assertEqual(fw_debug_info[0], ('B', 0))


class TestOutputSort(unittest.TestCase):
    def test_get_sorted_outputs(self):
        nodes = {'A': {'type': 'Identity', 'kind': 'op'},
                 'B': {'type': 'Identity', 'kind': 'op'},
                 'C': {'type': 'Identity', 'kind': 'op'},
                 'D': {'type': 'Identity', 'kind': 'op'},
                 'E': {'type': 'Identity', 'kind': 'op'},
                 'F': {'type': 'Identity', 'kind': 'op'},
                 'G': {'type': 'Identity', 'kind': 'op'},
                 'H': {'type': 'Identity', 'kind': 'op'},
                 'Ad': {'value': None, 'kind': 'data'},
                 'Bd': {'value': None, 'kind': 'data'},
                 'Cd': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('C', 0)]},
                 'Dd': {'value': None, 'kind': 'data'},
                 'Ed': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('E', 0)]},
                 'Fd': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('F', 0)]},
                 'Gd': {'value': None, 'kind': 'data'},
                 'Hd': {'value': None, 'kind': 'data'}
                 }
        edges = [
            ('A', 'Ad', {'out': 0}),
            ('Ad', 'B', {'in': 0}),
            ('B', 'Bd', {'out': 0}),
            ('Bd', 'C', {'in': 0}),
            ('C', 'Cd', {'out': 0}),
            ('Cd', 'D', {'in': 0}),
            ('D', 'Dd', {'out': 0}),
            ('Dd', 'E', {'in': 0}),
            ('E', 'Ed', {'out': 0}),
            ('Cd', 'F', {'in': 0}),
            ('F', 'Fd', {'out': 0}),
            ('Fd', 'G', {'in': 0}),
            ('G', 'Gd', {'out': 0}),
            ('Cd', 'H', {'in': 0}),
            ('H', 'Hd', {'out': 0})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        self.assertListEqual([node.id for node in get_sorted_outputs(graph)], ['Hd', 'Ed', 'Gd'])

    def test_get_sorted_outputs_fine_situation(self):
        nodes = {'A': {'type': 'Identity', 'kind': 'op'},
                 'B': {'type': 'Identity', 'kind': 'op'},
                 'C': {'type': 'Identity', 'kind': 'op'},
                 'D': {'type': 'Identity', 'kind': 'op'},
                 'E': {'type': 'Identity', 'kind': 'op'},
                 'F': {'type': 'Identity', 'kind': 'op'},
                 'G': {'type': 'Identity', 'kind': 'op'},
                 'H': {'type': 'Identity', 'kind': 'op'},
                 'Ad': {'value': None, 'kind': 'data'},
                 'Bd': {'value': None, 'kind': 'data'},
                 'Cd': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('C', 0)]},
                 'Dd': {'value': None, 'kind': 'data'},
                 'Ed': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('E', 0)]},
                 'Fd': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('F', 0)]},
                 'Gd': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('G', 0)]},
                 'Hd': {'value': None, 'kind': 'data', 'fw_tensor_debug_info': [('H', 0)]}
                 }
        edges = [
            ('A', 'Ad', {'out': 0}),
            ('Ad', 'B', {'in': 0}),
            ('B', 'Bd', {'out': 0}),
            ('Bd', 'C', {'in': 0}),
            ('C', 'Cd', {'out': 0}),
            ('Cd', 'D', {'in': 0}),
            ('D', 'Dd', {'out': 0}),
            ('Dd', 'E', {'in': 0}),
            ('E', 'Ed', {'out': 0}),
            ('Cd', 'F', {'in': 0}),
            ('F', 'Fd', {'out': 0}),
            ('Fd', 'G', {'in': 0}),
            ('G', 'Gd', {'out': 0}),
            ('Cd', 'H', {'in': 0}),
            ('H', 'Hd', {'out': 0})
        ]
        graph = build_graph_with_edge_attrs(nodes, edges)
        self.assertListEqual([node.id for node in get_sorted_outputs(graph)], ['Ed', 'Gd', 'Hd'])
