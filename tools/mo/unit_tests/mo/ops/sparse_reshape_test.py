# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.sparse_reshape import SparseReshape
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import valued_const_with_data, result, regular_op_with_empty_data, connect, \
    shaped_parameter, build_graph, empty_data

dyn = dynamic_dimension


class TestSparseReshape(UnitTestWithMockedTelemetry):

    def build_and_test_shape_inference(self, input_indices_sparse_shape, input_actual_shape, new_shape, ref_out_shape,
                                       input_indices=None, ref_out_indices=None):
        # sparse tensor is stored in COO format
        nodes = {
            **shaped_parameter('input_indices', shape_array(input_indices_sparse_shape), {'value': input_indices}),
            **valued_const_with_data('input_shape', shape_array(input_actual_shape)),
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

        graph = build_graph(nodes, edges, update_attributes={'input_indices_d': {'value': input_indices}})
        graph.stage = 'middle'
        partial_infer(graph)

        node = Node(graph, 'sparse_reshape_node')
        output_indices = node.out_port(0).data.get_value()
        actual_output_shape = node.out_port(1).data.get_value()
        self.assertTrue(strict_compare_tensors(actual_output_shape, ref_out_shape))
        self.assertTrue(strict_compare_tensors(output_indices, ref_out_indices))

    def test_sparse_shape_1(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[4, 5],
                                            new_shape=[5, -1, 2],
                                            ref_out_shape=[5, 2, 2])

    def test_sparse_shape_2(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[dyn, 5, 6],
                                            new_shape=[5, -1],
                                            ref_out_shape=[5, dyn])

    def test_sparse_shape_3(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[5, 3, 8],
                                            new_shape=[4, dyn],
                                            ref_out_shape=[4, 30])

    def test_sparse_shape_4(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[1, 30],
                                            new_shape=[1, dyn],
                                            ref_out_shape=[1, 30])

    def test_sparse_shape_5(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[1, 30],
                                            new_shape=[3, dyn, dyn],
                                            ref_out_shape=[3, dyn, dyn])

    def test_sparse_shape_6(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[1, 30],
                                            new_shape=[dyn, 3, dyn],
                                            ref_out_shape=[dyn, 3, dyn])

    def test_sparse_shape_7(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[dyn, 30],
                                            new_shape=[dyn, dyn, 33],
                                            ref_out_shape=[dyn, dyn, 33])

    def test_sparse_shape_8(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[dyn, 30],
                                            new_shape=[dyn, 3, -1],
                                            ref_out_shape=[dyn, 3, dyn])

    def test_sparse_shape_9(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        self.build_and_test_shape_inference(input_indices_sparse_shape=[11, 2],
                                            input_actual_shape=[dyn, 30],
                                            new_shape=[1, dyn],
                                            ref_out_shape=[1, dyn])

    def test_sparse_shape_10(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        sparse_shape = [11, 2]
        input_indices_value = np.arange(0, np.prod(sparse_shape)).reshape(sparse_shape)
        self.build_and_test_shape_inference(input_indices_sparse_shape=sparse_shape,
                                            input_actual_shape=[1, 30],
                                            new_shape=[1, dyn],
                                            ref_out_shape=[1, 30],
                                            input_indices=input_indices_value,
                                            ref_out_indices=input_indices_value)

    def test_sparse_shape_11(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        sparse_shape = [11, 2]
        self.build_and_test_shape_inference(input_indices_sparse_shape=sparse_shape,
                                            input_actual_shape=[1, 30],
                                            new_shape=[1, 15, 2],
                                            ref_out_shape=[1, 15, 2])

    # negative test with uncompatible shapes
    def test_sparse_shape_12(self):
        # ref_output_indices_shape = np.array([11, 3], dtype=np.int32)
        sparse_shape = [11, 2]
        with self.assertRaisesRegex(Error, 'Stopped shape/value propagation'):
            self.build_and_test_shape_inference(input_indices_sparse_shape=sparse_shape,
                                                input_actual_shape=[1, 30],
                                                new_shape=[1, 64, 2],
                                                ref_out_shape=[1, 15, 2])
