# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.elementwise import Div
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

graph_nodes_attrs = {
    'A': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'A_data': {'kind': 'data', 'shape': None, 'value': None},
    'B': {'type': 'Const', 'op': 'Const', 'kind': 'op', 'shape': None, 'value': None},
    'B_data': {'kind': 'data', 'shape': None, 'value': None, 'dim_attrs': []},
    'div': {'type': 'Divide', 'op': 'Div', 'kind': 'op'},
    'div_data': {'kind': 'data', 'value': None, 'shape': None},
    'output': {'kind': 'op', 'op': 'Result'},
}


graph_edges = [
    ('A', 'A_data'),
    ('B', 'B_data'),
    ('A_data', 'div', {'in': 0}),
    ('B_data', 'div', {'in': 1}),
    ('div', 'div_data'),
    ('div_data', 'output'),
]


class TestDivValuePropagation():
    @pytest.mark.parametrize("a_shape, a_value, b_shape, b_value, elem_type",[
        ([2, 3], np.array([[1, 4, -6], [0, -16, 45]], dtype=np.int64),
         [2, 3], np.array([[1, 2, -4], [1, -8, -5]], dtype=np.int64),
         np.int64),
        ([2, 3], np.array([[1, 4, -6], [0, -16, 45]], dtype=np.int64),
         [2, 3], np.array([[1, 2, -4], [1, -8, -5]], dtype=np.int64),
         np.float64),
        ([2, 3], np.array([[1, 4, -6], [0, -16, 45]], dtype=np.int64),
         [2, 3], np.array([[1, 2, -4], [1, -8, -5]], dtype=np.int64),
         np.float32),
        ([3, 3], np.array([[15, 2, 11], [14, 7, 8], [24, 12, 0]], dtype=np.int64),
         [3, 3], np.array([[-5, 4, 2], [7, 2, 4], [6, 24, 1]], dtype=np.int64),
         np.int64),
        ([3, 3], np.array([[15, 2, 11], [14, 7, 8], [24, 12, 0]], dtype=np.int64),
         [3, 3], np.array([[-5, 4, 2], [7, 2, 4], [6, 24, 1]], dtype=np.int64),
         np.float64),
        ([3, 3], np.array([[15, 2, 11], [14, 7, 8], [24, 12, 0]], dtype=np.int64),
         [3, 3], np.array([[-5, 4, 2], [7, 2, 4], [6, 24, 1]], dtype=np.int64),
         np.float32),
    ])
    def test_value_propagation(self, a_shape, a_value, b_shape, b_value, elem_type):
        graph = build_graph(
            nodes_attrs=graph_nodes_attrs,
            edges=graph_edges,
            update_attributes={
                'A': {'shape': int64_array(a_shape), 'value': a_value.astype(elem_type)},
                'A_data': {'shape': int64_array(a_shape), 'value': a_value.astype(elem_type)},
                'B': {'shape': int64_array(b_shape), 'value': b_value.astype(elem_type)},
                'B_data': {'shape': int64_array(b_shape), 'value': b_value.astype(elem_type)},
            }
        )
        node = Node(graph, 'div')
        node['infer'] = Div(graph, node.attrs()).create_node().infer
        node.infer(node)
        node_data = node.out_port(0).get_destination().data.get_value()

        def func_for_ref():
            if np.issubdtype(elem_type, np.integer):
                return lambda a, b: a // b
            else:
                return lambda a, b: a / b

        ref_data = func_for_ref()(a_value, b_value)
        node_data_shape = node_data.shape
        ref_data_shape = ref_data.shape
        msg = "Value propagation for 'div' node is not correct."
        assert node_data_shape == ref_data_shape and np.all(node_data == ref_data), msg
