# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy.testing as npt

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, strict_compare_tensors, \
    dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.ops.gather import Gather
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from unit_tests.utils.graph import valued_const_with_data, result, regular_op_with_empty_data, connect, \
    shaped_parameter, build_graph


class TestGatherPartialInfer(UnitTestWithMockedTelemetry):

    @staticmethod
    def build_and_test_value_inference(data, indices, axis, batch_dims, ref_value, negative_test_string=None):
        nodes = {
            **valued_const_with_data('data', int64_array(data)),
            **valued_const_with_data('indices', int64_array(indices)),
            **valued_const_with_data('axis', int64_array(axis)),
            **regular_op_with_empty_data('gather', {'op': 'Gather', 'batch_dims': batch_dims, 'infer': Gather.infer}),
            **result('res'),
        }

        edges = [
            *connect('data', '0:gather'),
            *connect('indices', '1:gather'),
            *connect('axis', '2:gather'),
            *connect('gather', 'res')
        ]

        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        partial_infer(graph)

        node = Node(graph, 'gather')
        res = node.out_port(0).data.get_value()
        npt.assert_array_equal(res, ref_value)

    @staticmethod
    def build_and_test_shape_inference(data_shape, indices_shape, axis, batch_dims, ref_shape):
        nodes = {
            **shaped_parameter('data', int64_array(data_shape)),
            **shaped_parameter('indices', int64_array(indices_shape)),
            **valued_const_with_data('axis', int64_array(axis)),
            **regular_op_with_empty_data('gather', {'op': 'Gather', 'batch_dims': batch_dims, 'infer': Gather.infer}),
            **result('res'),
        }

        edges = [
            *connect('data', '0:gather'),
            *connect('indices', '1:gather'),
            *connect('axis', '2:gather'),
            *connect('gather', 'res')
        ]

        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        partial_infer(graph)

        node = Node(graph, 'gather')
        res = node.out_port(0).data.get_shape()
        npt.assert_array_equal(res, ref_shape)

    def test_shape_axis_1(self):
        self.build_and_test_shape_inference(axis=1, batch_dims=0,
                                            data_shape=[3, 3],
                                            indices_shape=[1, 2],
                                            ref_shape=[3, 1, 2])

    def test_shape_axis_1_1(self):
        self.build_and_test_shape_inference(axis=1, batch_dims=0,
                                            data_shape=[3, 3],
                                            indices_shape=[1, 2, 4],
                                            ref_shape=[3, 1, 2, 4])

    def test_shape_axis_1_2(self):
        self.build_and_test_shape_inference(axis=1, batch_dims=0,
                                            data_shape=[1, 2, 4],
                                            indices_shape=[3, 3],
                                            ref_shape=[1, 3, 3, 4])

    def test_shape_axis_1_3(self):
        self.build_and_test_shape_inference(axis=1, batch_dims=0,
                                            data_shape=[1, 2, 4],
                                            indices_shape=[5, 8, 16],
                                            ref_shape=[1, 5, 8, 16, 4])

    def test_shape_axis_0(self):
        self.build_and_test_shape_inference(axis=0, batch_dims=0,
                                            data_shape=[3, 3],
                                            indices_shape=[1, 2],
                                            ref_shape=[1, 2, 3])

    def test_shape_axis_0_1(self):
        self.build_and_test_shape_inference(axis=0, batch_dims=0,
                                            data_shape=[3, 3],
                                            indices_shape=[1, 2, 5],
                                            ref_shape=[1, 2, 5, 3])

    def test_shape_axis_0_2(self):
        self.build_and_test_shape_inference(axis=0, batch_dims=0,
                                            data_shape=[1, 2, 5],
                                            indices_shape=[3, 3],
                                            ref_shape=[3, 3, 2, 5])

    def test_shape_axis_0_3(self):
        self.build_and_test_shape_inference(axis=0, batch_dims=0,
                                            data_shape=[1, 2, 5],
                                            indices_shape=[6, 8, 15],
                                            ref_shape=[6, 8, 15, 2, 5])

    def test_shape_axis_minus_2(self):
        self.build_and_test_shape_inference(axis=-2, batch_dims=0,
                                            data_shape=[2, 3, 7],
                                            indices_shape=[1, 4],
                                            ref_shape=[2, 1, 4, 7])

    def test_shape_axis_1_batch_dims_1(self):
        self.build_and_test_shape_inference(axis=1, batch_dims=1,
                                            data_shape=[3, 4],
                                            indices_shape=[3, 1, 2],
                                            ref_shape=[3, 1, 2])

    def test_shape_axis_2_batch_dims_1(self):
        self.build_and_test_shape_inference(axis=2, batch_dims=1,
                                            data_shape=[3, 4, 7],
                                            indices_shape=[3, 1, 2],
                                            ref_shape=[3, 4, 1, 2])

    def test_shape_axis_2_batch_dims_minus_1(self):
        self.build_and_test_shape_inference(axis=2, batch_dims=-1,
                                            data_shape=[3, 1, 7],
                                            indices_shape=[3, 1, 2],
                                            ref_shape=[3, 1, 2])

    def test_shape_axis_2_batch_dims_minus_2(self):
        self.build_and_test_shape_inference(axis=2, batch_dims=-2,
                                            data_shape=[3, 4, 7],
                                            indices_shape=[3, 1, 2],
                                            ref_shape=[3, 4, 1, 2])

    def test_axis_0_batch_dims_0(self):
        self.build_and_test_value_inference(axis=0, batch_dims=0,
                                            data=[1, 2, 3, 4, 5],
                                            indices=[0, 0, 4],
                                            ref_value=[1, 1, 5])

    def test_axis_0_batch_dims_0_negative_indices(self):
        self.build_and_test_value_inference(axis=0, batch_dims=0,
                                            data=[1, 2, 3, 4, 5],
                                            indices=[-1, -2, -3],
                                            ref_value=[5, 4, 3])

    def test_axis_1_batch_dims_1(self):
        self.build_and_test_value_inference(axis=1, batch_dims=1,
                                            data=[[1, 2, 3, 4, 5],
                                                  [6, 7, 8, 9, 10]],
                                            indices=[[0, 0, 4],
                                                     [4, 0, 0]],

                                            ref_value=[[1, 1, 5],
                                                       [10, 6, 6]])

    def test_axis_minus_1_batch_dims_1(self):
        self.build_and_test_value_inference(axis=-1, batch_dims=1,
                                            data=[[1, 2, 3, 4, 5],
                                                  [6, 7, 8, 9, 10]],
                                            indices=[[0, 0, 4],
                                                     [4, 0, 0]],

                                            ref_value=[[1, 1, 5],
                                                       [10, 6, 6]])

    def test_axis_2_batch_dims_1(self):
       self.build_and_test_value_inference(axis=2, batch_dims=1,
                                           data=[[[[ 1,  2,  3,  4],  # <-- first batch
                                                   [ 5,  6,  7,  8],
                                                   [ 9, 10, 11, 12],
                                                   [13, 14, 15, 16],
                                                   [17, 18, 19, 20]]],
                                                 [[[21, 22, 23, 24],  # < -- second batch
                                                   [25, 26, 27, 28],
                                                   [29, 30, 31, 32],
                                                   [33, 34, 35, 36],
                                                   [37, 38, 39, 40]]]],  # data_shape = (2, 1, 5, 4)
                                           indices=[[1, 2, 4],
                                                    [4, 3, 2]],
                                           ref_value=[[[[ 5,  6,  7,  8],
                                                        [ 9, 10, 11, 12],
                                                        [17, 18, 19, 20]]],
                                                      [[[37, 38, 39, 40],
                                                        [33, 34, 35, 36],
                                                        [29, 30, 31, 32]]]])

    def test_axis_2_batch_dims_1_with_negative_indices(self):
       self.build_and_test_value_inference(axis=2, batch_dims=1,
                                           data=[[[[ 1,  2,  3,  4],  # <-- first batch
                                                   [ 5,  6,  7,  8],
                                                   [ 9, 10, 11, 12],
                                                   [13, 14, 15, 16],
                                                   [17, 18, 19, 20]]],
                                                 [[[21, 22, 23, 24],  # < -- second batch
                                                   [25, 26, 27, 28],
                                                   [29, 30, 31, 32],
                                                   [33, 34, 35, 36],
                                                   [37, 38, 39, 40]]]],  # data_shape = (2, 1, 5, 4)
                                           indices=[[-4, -3, -1],
                                                    [-1, 3, 2]],
                                           ref_value=[[[[ 5,  6,  7,  8],
                                                        [ 9, 10, 11, 12],
                                                        [17, 18, 19, 20]]],
                                                      [[[37, 38, 39, 40],
                                                        [33, 34, 35, 36],
                                                        [29, 30, 31, 32]]]])

    def test_axis_2_batch_dims_mimus_1(self):
        self.build_and_test_value_inference(axis=2, batch_dims=-1,
                                            data=[[[[ 1,  2,  3,  4],  # <-- first batch
                                                    [ 5,  6,  7,  8],
                                                    [ 9, 10, 11, 12],
                                                    [13, 14, 15, 16],
                                                    [17, 18, 19, 20]]],
                                                  [[[21, 22, 23, 24],  # < -- second batch
                                                    [25, 26, 27, 28],
                                                    [29, 30, 31, 32],
                                                    [33, 34, 35, 36],
                                                    [37, 38, 39, 40]]]],  # data_shape = (2, 1, 5, 4)
                                            indices=[[1, 2, 4],
                                                     [4, 3, 2]],
                                            ref_value=[[[[ 5,  6,  7,  8],
                                                         [ 9, 10, 11, 12],
                                                         [17, 18, 19, 20]]],
                                                       [[[37, 38, 39, 40],
                                                         [33, 34, 35, 36],
                                                         [29, 30, 31, 32]]]])

    # negative tests
    def test_shape_indices_data_shape_inconsistency(self):
        self.assertRaises(Error, self.build_and_test_shape_inference,
                          axis=2, batch_dims=2,
                          data_shape=[3, 4, 7],
                          indices_shape=[3, 1, 2],
                          ref_shape=[3, 4, 2])

    def test_shape_batch_dims_greater_than_axis(self):
        self.assertRaises(Error, self.build_and_test_shape_inference,
                          axis=2, batch_dims=3,
                          data_shape=[3, 4, 7],
                          indices_shape=[3, 4, 2],
                          ref_shape=[3, 4, 2])

    def test_shape_batch_dims_out_of_bound(self):
        self.assertRaises(Error, self.build_and_test_shape_inference,
                          axis=2, batch_dims=4,
                          data_shape=[3, 4, 7],
                          indices_shape=[3, 4, 2],
                          ref_shape=[3, 4, 2])

    def test_shape_axis_out_of_bound(self):
        self.assertRaises(Error, self.build_and_test_shape_inference,
                          axis=3, batch_dims=2,
                          data_shape=[3, 4, 7],
                          indices_shape=[3, 4, 2],
                          ref_shape=[3, 4, 2])


dyn = dynamic_dimension_value


class TestElementwiseReverseInfer(UnitTestWithMockedTelemetry):
    @staticmethod
    def build_and_test_reverse_inference(data_shape, indices_shape, axis, batch_dims, out_shape, ref_shape):
        in_port_with_defined_shape = 0 if data_shape is not None else 1
        defined_shape = shape_array(data_shape if data_shape is not None else indices_shape)

        nodes = {
            **shaped_parameter('data', data_shape, {'reverse_infer': Parameter.reverse_infer}),
            **shaped_parameter('indices', indices_shape, {'reverse_infer': Parameter.reverse_infer}),
            **valued_const_with_data('axis', int64_array(axis)),
            **regular_op_with_empty_data('gather', {'op': 'Gather', 'batch_dims': batch_dims,
                                                    'infer': Gather.infer,
                                                    'reverse_infer': Gather.reverse_infer}),
            **result('res'),
        }

        edges = [
            *connect('data', '0:gather'),
            *connect('indices', '1:gather'),
            *connect('axis', '2:gather'),
            *connect('gather', 'res')
        ]

        graph = build_graph(nodes, edges)
        graph.stage = 'middle'

        Node(graph, 'gather').out_port(0).data.set_shape(shape_array(out_shape))
        Node(graph, 'gather').in_port(in_port_with_defined_shape).data.set_shape(defined_shape)

        partial_infer(graph)
        actual_shape = Node(graph, 'gather').in_port(int(not in_port_with_defined_shape)).data.get_shape()
        assert strict_compare_tensors(actual_shape, shape_array(ref_shape))

    # undefined indices pshape
    def test_reverse_infer_1(self):
        self.build_and_test_reverse_inference(data_shape=[dyn, dyn],
                                              indices_shape=None,
                                              axis=0,
                                              batch_dims=0,
                                              out_shape=[dyn, dyn, dyn, dyn],
                                              ref_shape=[dyn, dyn, dyn])

    def test_reverse_infer_2(self):
        self.build_and_test_reverse_inference(data_shape=[3, 10],
                                              indices_shape=None,
                                              axis=1,
                                              batch_dims=0,
                                              out_shape=[3, 40, 50, 60],
                                              ref_shape=[40, 50, 60])

    # undefined data pshape
    def test_reverse_infer_3(self):
        self.build_and_test_reverse_inference(data_shape=None,
                                              indices_shape=[4, 5],
                                              axis=0,
                                              batch_dims=0,
                                              out_shape=[4, 5, 10],
                                              ref_shape=[dyn, 10])

    def test_reverse_infer_4(self):
        self.build_and_test_reverse_inference(data_shape=None,
                                              indices_shape=[4, 67],
                                              axis=1,
                                              batch_dims=0,
                                              out_shape=[3, 4, 67, 100],
                                              ref_shape=[3, dyn, 100])
