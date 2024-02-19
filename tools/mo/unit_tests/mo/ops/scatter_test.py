# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.scatter import ScatterElementsUpdate, ScatterUpdate
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, dynamic_dimension_value
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, valued_const_with_data


class TestScatterElementsInferTest():
    @pytest.mark.parametrize("data, indices, updates, axis, ref_res",[
        ([[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],
         [[1, 0, 2],
          [0, 2, 1]],
         [[1.0, 1.1, 1.2],
          [2.0, 2.1, 2.2]],
         0,
         [[2.0, 1.1, 0.0],
          [1.0, 0.0, 2.2],
          [0.0, 2.1, 1.2]]),

        ([[1.0, 2.0, 3.0, 4.0, 5.0]],
         [[1, 3]],
         [[1.1, 2.1]],
         1,
         [[1.0, 1.1, 3.0, 2.1, 5.0]]),

        ([[1.0, 2.0, 3.0, 4.0, 5.0]],
         [[1, 3]],
         [[1.1, 2.1]],
         [1],
         [[1.0, 1.1, 3.0, 2.1, 5.0]]),

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
             [[1, 0],
              [1, 0]],
             [[0, 1],
              [1, 0]]
         ],
         [
             [[21, 22],
              [23, 24]],
             [[25, 26],
              [27, 28]],
             [[29, 30],
              [31, 32]]
         ],
         -1,  # axis
         [
             [[22, 21],
              [23, 24]],
             [[26, 25],
              [28, 27]],
             [[29, 30],
              [32, 31]]
         ]),
    ])
    def test_scatterelements_value_infer(self, data, indices, updates, axis, ref_res):
        nodes = {
            **valued_const_with_data('data', np.array(data)),
            **valued_const_with_data('indices', int64_array(indices)),
            **valued_const_with_data('updates', np.array(updates)),
            **valued_const_with_data('axis', int64_array(axis)),
            **regular_op_with_empty_data('scatter_elements', {'op': 'ScatterElementsUpdate', 'axis': axis}),
            **result()
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('data', '0:scatter_elements'),
            *connect('indices', '1:scatter_elements'),
            *connect('updates', '2:scatter_elements'),
            *connect('axis', '3:scatter_elements'),
            *connect('scatter_elements', 'output')
        ], nodes_with_edges_only=True)
        graph.stage = 'middle'

        scatter_el_node = Node(graph, 'scatter_elements')
        ScatterElementsUpdate.infer(scatter_el_node)

        res_output_shape = scatter_el_node.out_node().shape
        assert np.array_equal(int64_array(ref_res).shape, res_output_shape)

        res_output_value = scatter_el_node.out_node().value
        assert np.array_equal(ref_res, res_output_value)


class TestScatterUpdateInferTest():
    @pytest.mark.parametrize("data, indices, updates, axis, ref_res",[
        ([[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],
         [[1, 2]],
         [[[1.0, 1.1, 1.2],
           [2.0, 2.1, 2.2]]],
         0,
         [[0.0, 0.0, 0.0],
          [1.0, 1.1, 1.2],
          [2.0, 2.1, 2.2]]),

        # negative axis
        ([[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],
         [[1, 2]],
         [[[1.0, 1.1]],
          [[1.2, 2.0]],
          [[2.1, 2.2]]],
         -1,
         [[0.0, 1.0, 1.1],
          [0.0, 1.2, 2.0],
          [0.0, 2.1, 2.2]]),

        # one element
        ([[[0., 0.], [0., 0.], [0., 0.]],
          [[0., 0.], [0., 0.], [0., 0.]],
          [[0., 0.], [0., 0.], [0., 0.]]],
         [[1]],
         [[[[1., 2.], [3., 4.], [5., 6.]]]],
         0,
         [[[0., 0.], [0., 0.], [0., 0.]],
          [[1., 2.], [3., 4.], [5., 6.]],
          [[0., 0.], [0., 0.], [0., 0.]]]),

        # shape [2,3,3]
        ([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
          [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]],
         # indices [3,2]
         [[1, 2], [0, 1], [1, 2]],
         # updates [2,3,2,3]
         [[[[1., 2., 3.], [4., 5., 6.]],
           [[7., 8., 9.], [9., 8., 7.]],
           [[6., 5., 4.], [3., 2., 1.]]],
          [[[1., 2., 3.], [4., 5., 6.]],
           [[7., 8., 9.], [9., 8., 7.]],
           [[6., 5., 4.], [3., 2., 1.]]]],
         # axis
         1,
         # ref
         [[[7., 8., 9.], [6., 5., 4.], [3., 2., 1.]],
          [[7., 8., 9.], [6., 5., 4.], [3., 2., 1.]]]),

        # dynamic updates
        ([0, 0, 0],
         [2],
         shape_array([dynamic_dimension_value]),
         0,
         shape_array([0, 0, dynamic_dimension_value])),
    ])
    def test_scatter_update_value_infer(self, data, indices, updates, axis, ref_res):
        nodes = {
            **valued_const_with_data('data', np.array(data)),
            **valued_const_with_data('indices', int64_array(indices)),
            **valued_const_with_data('updates', np.array(updates)),
            **valued_const_with_data('axis', int64_array(axis)),
            **regular_op_with_empty_data('scatter_update', {'op': 'ScatterUpdate', 'axis': axis}),
            **result()
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('data', '0:scatter_update'),
            *connect('indices', '1:scatter_update'),
            *connect('updates', '2:scatter_update'),
            *connect('axis', '3:scatter_update'),
            *connect('scatter_update', 'output')
        ], nodes_with_edges_only=True)
        graph.stage = 'middle'

        scatter_update_node = Node(graph, 'scatter_update')
        ScatterUpdate.infer(scatter_update_node)

        res_output_shape = scatter_update_node.out_node().shape
        assert np.array_equal(int64_array(ref_res).shape, res_output_shape)

        res_output_value = scatter_update_node.out_node().value
        assert np.array_equal(ref_res, res_output_value)
