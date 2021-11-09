# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import numpy.testing as npt
from extensions.ops.sparse_reshape import SparseReshape
from mo.front.common.partial_infer.utils import int64_array, shape_array, compatible_shapes, dynamic_dimension
from mo.graph.graph import Node
from mo.middle.passes.infer import partial_infer
from mo.utils.error import Error
from unit_tests.utils.graph import valued_const_with_data, result, regular_op_with_empty_data, connect, \
    shaped_parameter, build_graph, empty_data

dyn = dynamic_dimension

class TestSparseReshape(unittest.TestCase):

    @staticmethod
    def build_and_test_shape_inference(input_indices_sparse_shape, input_shape, new_shape, ref_out_shape):
        nodes = {
            **shaped_parameter('input_indices', shape_array(input_indices_sparse_shape)),
            **valued_const_with_data('input_shape', shape_array(input_shape)),
            **valued_const_with_data('new_shape', shape_array(new_shape)),
            **regular_op_with_empty_data('sparse_reshape_node', {'op': 'SparseReshape',
                                                                 'special_zero': True,
                                                                 'infer': SparseReshape.infer}),
            **empty_data('sparse_reshape_node_d:out_port_1'),

            **result('output_indices'),
            **result('output_shape'),
        }

        edges = [
            *connect('input_indices', '0:sparse_reshape_node'),
            *connect('input_shape', '1:sparse_reshape_node'),
            *connect('new_shape', '2:sparse_reshape_node'),
            *connect('sparse_reshape_node:0', 'output_indices'),
            ('sparse_reshape_node', 'sparse_reshape_node_d:out_port_1', {'out': 1}),
            ('sparse_reshape_node_d:out_port_1', 'output_shape', {'in': 0}),
        ]

        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        partial_infer(graph)

        node = Node(graph, 'sparse_reshape_node')
        output_indices = node.out_port(0).data.get_value()
        actual_output_shape = node.out_port(1).data.get_value()
        npt.assert_equal(actual_output_shape, ref_out_shape)

    def test_static_shape_1(self):
        # ref_output_indices_shape = np.array([5, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[5, 2],
                                            input_shape=[4, 5],
                                            new_shape=[5, -1, 2],
                                            ref_out_shape=[5, 2, 2])

    def test_static_shape_2(self):
        # ref_output_indices_shape = np.array([5, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[5, 2],
                                            input_shape=[dyn, 5, 6],
                                            new_shape=[0, -1],
                                            ref_out_shape=[dyn, 30])

    def test_static_shape_3(self):
        # ref_output_indices_shape = np.array([5, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[5, 2],
                                            input_shape=[5, 3, 8],
                                            new_shape=[4, dyn],
                                            ref_out_shape=[4, 30])
