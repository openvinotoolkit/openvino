# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.gatherelements import GatherElements
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, strict_compare_tensors, dynamic_dimension
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, \
    valued_const_with_data, shaped_parameter

dyn = dynamic_dimension

class TestGatherElementsInferTest():
    @pytest.mark.parametrize("data, indices, axis, ref_res",[
        ([[1, 2],
          [3, 4]],
         [[0, 1],
          [0, 0]],
         0,  # axis
         [[1, 4],  # ref_res
          [1, 2]]),

        ([[1, 2],
          [3, 4]],
         [[0, 1],
          [0, 0]],
         1,  # axis
         [[1, 2],  # ref_res
          [3, 3]]),

        ([[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]],
         [[1, 2, 0],
          [2, 0, 0]],
         0,  # axis
         [[4, 8, 3],  # ref_res
          [7, 2, 3]]),

        ([[1, 2],
          [3, 4]],
         [[0, 1],
          [0, 0]],
         -1,  # axis
         [[1, 2],  # ref_res
          [3, 3]]),

        ([  # 3D case
             [[1, 2],
              [3, 4]],
             [[5, 6],
              [7, 8]],
             [[9, 10],
              [11, 12]]
         ],
         [
             [[1, 0],
              [0, 1]],
             [[1, 1],
              [1, 0]],
             [[0, 0],
              [1, 1]]
         ],
         -1,  # axis
         [
             [[2, 1],
              [3, 4]],
             [[6, 6],
              [8, 7]],
             [[9, 9],
              [12, 12]]
         ]),
    ])
    def test_gatherelements_value_infer(self, data, indices, axis, ref_res):
        nodes = {
            **valued_const_with_data('data', int64_array(data)),
            **valued_const_with_data('indices', int64_array(indices)),
            **regular_op_with_empty_data('gather_elements', {'op': 'GatherElements', 'axis': axis}),
            **result()
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('data', '0:gather_elements'),
            *connect('indices', '1:gather_elements'),
            *connect('gather_elements', 'output')
        ], nodes_with_edges_only=True)
        graph.stage = 'middle'

        gather_el_node = Node(graph, 'gather_elements')
        GatherElements.infer(gather_el_node)

        res_output_shape = gather_el_node.out_node().shape
        assert np.array_equal(int64_array(ref_res).shape, res_output_shape)

        res_output_value = gather_el_node.out_node().value
        if res_output_value is not None:
            assert np.array_equal(int64_array(ref_res), res_output_value)

    def check_shape_infer(self, data_shape, indices_shape, axis, ref):
        nodes = {
            **shaped_parameter('data', data_shape),
            **shaped_parameter('indices', indices_shape),
            **regular_op_with_empty_data('gather_elements', {'op': 'GatherElements', 'axis': axis}),
            **result()
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('data', '0:gather_elements'),
            *connect('indices', '1:gather_elements'),
            *connect('gather_elements', 'output')
        ], nodes_with_edges_only=True)
        graph.stage = 'middle'

        gather_el_node = Node(graph, 'gather_elements')
        GatherElements.infer(gather_el_node)

        res_output_shape = gather_el_node.out_node().shape
        assert strict_compare_tensors(res_output_shape, ref)

    def test_shape_infer_1(self):
        self.check_shape_infer(data_shape=[3], indices_shape=[100], ref=[100], axis=0)

    def test_shape_infer_2(self):
        self.check_shape_infer(data_shape=[100, 4], indices_shape=[4, 4], ref=[4, 4], axis=0)

    def test_shape_infer_3(self):
        self.check_shape_infer(data_shape=[3, 4], indices_shape=[3, 100], ref=[3, 100], axis=1)

    def test_shape_infer_4(self):
        self.check_shape_infer(data_shape=[1, 3, 256], indices_shape=[256, 3, 256], ref=[256, 3, 256], axis=0)

    def test_shape_infer_5(self):
        self.check_shape_infer(data_shape=[1, 3, 256], indices_shape=[1, 1024, 256], ref=[1, 1024, 256], axis=1)

    def test_shape_infer_6(self):
        self.check_shape_infer(data_shape=[1, 3, 256], indices_shape=[1, 3, 1024], ref=[1, 3, 1024], axis=2)

    def test_shape_infer_7(self):
        self.check_shape_infer(data_shape=[1, 25, 64, 256], indices_shape=[1, 25, 64, 2], ref=[1, 25, 64, 2], axis=-1)

    # dynamic dimensions
    def test_shape_infer_8(self):
        self.check_shape_infer(data_shape=[dyn, 4], indices_shape=[3, 100], ref=[3, 100], axis=1)

    def test_shape_infer_9(self):
        self.check_shape_infer(data_shape=[100, 4], indices_shape=[dyn, 100], ref=[100, 100], axis=1)

    def test_shape_infer_10(self):
        self.check_shape_infer(data_shape=[1, 3, 256], indices_shape=[dyn, 3, 256], ref=[dyn, 3, 256], axis=0)

    def test_shape_infer_11(self):
        self.check_shape_infer(data_shape=[dyn, dyn, dyn], indices_shape=[dyn, dyn, dyn], ref=[dyn, dyn, dyn], axis=0)

    def test_shape_infer_12(self):
        self.check_shape_infer(data_shape=[1, 3, 256], indices_shape=[dyn, 1024, dyn], ref=[1, 1024, 256], axis=1)

    def test_shape_infer_13(self):
        self.check_shape_infer(data_shape=[1, 3, 256], indices_shape=[dyn, dyn, 1024], ref=[1, 3, 1024], axis=2)

    # negative tests
    def test_negative_shape_infer_ranks_differ(self):
        with pytest.raises(AssertionError):
            self.check_shape_infer(data_shape=[1, 3, 64], indices_shape=[1, 3], ref=[1, 3, 1024], axis=2)

    def test_negative_shape_infer_axis_out_of_bound(self):
        with pytest.raises(AssertionError):
            self.check_shape_infer(data_shape=[1, 4, 64], indices_shape=[1, 3, 64], ref=[1, 3, 1024], axis=20)

    def test_negative_shape_infer_inconsistent_shapes(self):
        with pytest.raises(Error):
            self.check_shape_infer(data_shape=[1, 4, 64], indices_shape=[1, 3, 64], ref=[1, 3, 1024], axis=2)
