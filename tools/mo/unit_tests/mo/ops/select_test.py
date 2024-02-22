# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.select import Select
from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension, shape_array, dynamic_dimension_value
from openvino.tools.mo.front.common.partial_infer.utils import strict_compare_tensors, int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph, valued_const_with_data, result, regular_op_with_empty_data, \
    connect


class TestSelect(unittest.TestCase):

    @staticmethod
    def build_select_graph_and_infer(condition_value, then_value, else_value, out_value,
                                     condition_shape=None, then_shape=None, else_shape=None, out_shape=None,
                                     auto_broadcast='numpy', fw_format=None):
        if then_value is not None:
            then_shape = int64_array(then_value.shape)
        if else_value is not None:
            else_shape = int64_array(else_value.shape)

        nodes = {
            **valued_const_with_data('then', then_value, then_shape),
            **valued_const_with_data('else', else_value, else_shape),
            **valued_const_with_data('condition', condition_value, condition_shape),
            **regular_op_with_empty_data('select', {'op': 'Select', 'auto_broadcast': auto_broadcast, 'format': fw_format}),
            **result('out'),
        }
        edges = [
            *connect('condition', '0:select'),
            *connect('then', '1:select'),
            *connect('else', '2:select'),
            *connect('select', 'out'),
        ]
        graph = build_graph(nodes, edges)

        select_node = Node(graph, 'select')
        Select.infer(select_node)

        select_out_node = Node(graph, 'select_d')

        value_desc = 'values'
        ref_val = out_value
        actual_val = select_out_node['value']
        if out_shape is not None:
            value_desc = 'shapes'
            ref_val = out_shape
            actual_val = select_out_node['shape']
            assert select_out_node['value'] is None, "if 'out_shape' is defined manually 'value' must be None"

        flag = strict_compare_tensors(actual_val, ref_val)
        msg = '' if flag else 'reference {} and actual {} {} do not match\n'.format(ref_val, actual_val, value_desc)
        return flag, msg

    def test_1(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([5, 6], dtype=bool),
                                                      then_value=np.ones([5, 6], dtype=float),
                                                      else_value=np.zeros([5, 6], dtype=float),
                                                      out_value=np.ones([5, 6], dtype=float))
        self.assertTrue(flag, msg)

    def test_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([15, 3, 5], dtype=bool),
                                                      then_value=np.ones([15, 3, 5], dtype=float),
                                                      else_value=np.zeros([15, 1, 5], dtype=float),
                                                      out_value=np.ones([15, 3, 5], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_no_condition(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=None, condition_shape=[2],
                                                      then_value=None, then_shape=[2],
                                                      else_value=None, else_shape=[2],
                                                      out_value=None, out_shape=[2])
        self.assertTrue(flag, msg)

    def test_select_infer_condition_true(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array([True, True], dtype=bool),
                                                      then_value=np.array([1, 1], dtype=np.int8),
                                                      else_value=np.array([2, 2], dtype=np.int8),
                                                      out_value=np.array([1, 1], dtype=np.int8))
        self.assertTrue(flag, msg)

    def test_select_infer_condition_false(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array([False, False], dtype=bool),
                                                      then_value=np.array([1, 1], dtype=np.int8),
                                                      else_value=np.array([2, 2], dtype=np.int8),
                                                      out_value=np.array([2, 2], dtype=np.int8))
        self.assertTrue(flag, msg)

    def test_select_infer_condition_true_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array([True], dtype=bool),
                                                      then_value=np.ones([15, 3, 5], dtype=float),
                                                      else_value=np.zeros([15, 1, 5], dtype=float),
                                                      out_value=np.ones([15, 3, 5], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_condition_true_then_and_else_are_scalars(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array([True], dtype=bool),
                                                      then_value=np.array(3, dtype=float),
                                                      else_value=np.array(1, dtype=float),
                                                      out_value=np.array([3], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_condition_true_then_and_else_are_scalars_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array(True, dtype=bool),
                                                      then_value=np.array(3, dtype=float),
                                                      else_value=np.array(1, dtype=float),
                                                      out_value=np.array(3, dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_condition_false_then_and_else_are_scalars(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array([False], dtype=bool),
                                                      then_value=np.array(3, dtype=float),
                                                      else_value=np.array(1, dtype=float),
                                                      out_value=np.array([1], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_condition_false_then_and_else_are_scalars_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array(False, dtype=bool),
                                                      then_value=np.array(3, dtype=float),
                                                      else_value=np.array(1, dtype=float),
                                                      out_value=np.array(1, dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_condition_false_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.array([False], dtype=bool),
                                                      then_value=np.ones([15, 3, 5], dtype=float),
                                                      else_value=np.zeros([15, 1, 5], dtype=float),
                                                      out_value=np.zeros([15, 3, 5], dtype=float))
        self.assertTrue(flag, msg)

    # if one of the branches is None then np.where shouldn't be used to avoid object dtype in output
    # res = np.where(condition, numpy_array_of_int[float]_dtype, None)
    # print(res.dtype) => object which is not compatible with other numeric dtypes, will fail further without
    # clear explanation, need to catch such cases as soon as possible
    def test_select_infer_None_then_branch_1(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.zeros([15, 3, 5], dtype=bool),
                                                      then_value=None, then_shape=[15, 3, 5],
                                                      else_value=np.ones([15, 1, 5], dtype=float),
                                                      out_value=np.ones([15, 3, 5], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_None_then_branch_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([15, 3, 5], dtype=bool),
                                                      then_value=None, then_shape=[15, 3, 5],
                                                      else_value=np.ones([15, 1, 5], dtype=float),
                                                      out_value=None)
        self.assertTrue(flag, msg)

    def test_select_infer_None_else_branch_1(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([15, 3, 5], dtype=bool),
                                                      then_value=np.ones([15, 1, 5], dtype=float),
                                                      else_value=None, else_shape=[15, 3, 5],
                                                      out_value=np.ones([15, 3, 5], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_infer_None_else_branch_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.zeros([15, 3, 5], dtype=bool),
                                                      then_value=np.ones([15, 1, 5], dtype=float),
                                                      else_value=None, else_shape=[15, 3, 5],
                                                      out_value=None)
        self.assertTrue(flag, msg)

    def test_select_broadcast_1(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([2, 3, 4, 5], dtype=bool),
                                                      then_value=np.ones([], dtype=float),
                                                      else_value=np.zeros([2, 3, 4, 5], dtype=float),
                                                      out_value=np.ones([2, 3, 4, 5], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_broadcast_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([2, 3, 4, 1], dtype=bool),
                                                      then_value= np.ones([1, 3, 1, 5], dtype=float),
                                                      else_value=np.zeros([2, 1, 1, 5], dtype=float),
                                                      out_value=np.ones([2, 3, 4, 5], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_broadcast_3(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([2, 3, 1, 1], dtype=bool),
                                                      then_value= np.ones([2, 3, 4, 5], dtype=float),
                                                      else_value=np.zeros([2, 1, 1, 5], dtype=float),
                                                      out_value=np.ones([2, 3, 4, 5], dtype=float))
        self.assertTrue(flag, msg)

    def test_select_broadcast_4(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ones([2, 3, 4, 5], dtype=bool),
                                                      then_value= np.ones([5], dtype=float),
                                                      else_value=np.zeros([2, 3, 4, 5], dtype=float),
                                                      out_value=np.ones([2, 3, 4, 5], dtype=float))
        self.assertTrue(flag, msg)

    # when output shape is broadcasted from condition, then, and else shapes
    def test_select_broadcast_with_shape(self):
        flag, msg = self.build_select_graph_and_infer(condition_shape=[2, 3, 4, 1], condition_value=None,
                                                      then_shape=[1, 3, 1, 5], then_value=None,
                                                      else_shape=[2, 1, 1, 5], else_value=None,
                                                      out_shape=[2, 3, 4, 5], out_value=None)
        self.assertTrue(flag, msg)

    def test_select_infer_assert_shapes(self):
        with self.assertRaisesRegex(AssertionError, "must be broadcastable"):
            self.build_select_graph_and_infer(condition_value=None, condition_shape=[2, 2],
                                              then_value=None, then_shape=[2, 2],
                                              else_value=None, else_shape=[3, 3],
                                              out_value=None, out_shape=[42, 42])

    def test_select_infer_assert_condition_shapes_are_compatible(self):
        with self.assertRaisesRegex(AssertionError, "must be broadcastable"):
            self.build_select_graph_and_infer(condition_value=None, condition_shape=[42, 3],
                                              then_value=None, then_shape=[1, 3],
                                              else_value=None, else_shape=[3, 3],
                                              out_value=None, out_shape=[3, 3])

    def test_select_infer_masked_1(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ma.array([True, True], mask=[1, 1]),
                                                      #  condition_value = [dynamic_dimension, dynamic_dimension])
                                                      then_value=None, then_shape=[2],
                                                      else_value=np.zeros((2, 2), dtype=np.int64),
                                                      out_value=None)
        self.assertTrue(flag, msg)

    def test_select_infer_masked_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ma.array([False, False], mask=[1, 1]),
                                                      # condition_value = [dynamic_dimension, dynamic_dimension])
                                                      then_value=None, then_shape=[2],
                                                      else_value=np.zeros((2, 2), dtype=np.int64),
                                                      out_value=None)
        self.assertTrue(flag, msg)

    def test_select_infer_masked_3(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ma.array([True, True], mask=[1, 1]),
                                                      # condition_value = [dynamic_dimension, dynamic_dimension])
                                                      then_value=None, then_shape=[2],
                                                      else_value=np.zeros((2, 2), dtype=np.int64),
                                                      out_value=None)
        self.assertTrue(flag, msg)

    def test_select_infer_masked_4(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ma.array([True, False], mask=[0, 1]),
                                                      #  condition_value = [True, dynamic_dimension])
                                                      then_value=np.ones((2, 2), dtype=np.int64),
                                                      else_value=np.zeros((2, 2), dtype=np.int64),
                                                      out_value=np.ma.array([[1, 42], [1, 42]], mask=[[0, 1], [0, 1]]))
                                                      # out_value = [[1, dynamic_dimension], [1, dynamic_dimension]]
        self.assertTrue(flag, msg)

    def test_select_infer_masked_5(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ma.array([False, True], mask=[0, 1]),
                                                      #  condition_value = [True, dynamic_dimension])
                                                      then_value=np.ones((2, 2), dtype=np.int64),
                                                      else_value=np.zeros((2, 2), dtype=np.int64),
                                                      out_value=np.ma.array([[0, 42], [0, 42]], mask=[[0, 1], [0, 1]]))
                                                      # out_value = [[0, dynamic_dimension], [0, dynamic_dimension]]
        self.assertTrue(flag, msg)

    def test_select_infer_masked_6(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=np.ma.array([True, False], mask=[1, 0]),
                                                      #  condition_value = [True, dynamic_dimension])
                                                      then_value=np.ones((2, 2), dtype=np.int64),
                                                      else_value=np.zeros((2, 2), dtype=np.int64),
                                                      out_value=np.ma.array([[42, 0], [42, 0]], mask=[[1, 0], [1, 0]]))
                                                      # out_value = [[dynamic_dimension, 0], [dynamic_dimension, 0]]
        self.assertTrue(flag, msg)

    def test_select_infer_no_broadcast_dynamic_then_else_shapes(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=None, condition_shape=shape_array([100, 100]),
                                                      then_value=None, then_shape=shape_array([100, dynamic_dimension_value]),
                                                      else_value=None, else_shape=shape_array([dynamic_dimension_value, 100]),
                                                      out_value=None, out_shape=shape_array([100, 100]),
                                                      auto_broadcast='none')
        self.assertTrue(flag, msg)

    def test_select_infer_no_broadcast_dynamic_then_else_shapes_2(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=None, condition_shape=shape_array([100, 100]),
                                                      then_value=None, then_shape=shape_array([dynamic_dimension_value, 100]),
                                                      else_value=None, else_shape=shape_array([100, dynamic_dimension_value]),
                                                      out_value=None, out_shape=shape_array([100, 100]),
                                                      auto_broadcast='none')
        self.assertTrue(flag, msg)

    def test_select_infer_no_broadcast_dynamic_shapes(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=None, condition_shape=shape_array([100, 100]),
                                                      then_value=None, then_shape=shape_array([100, dynamic_dimension_value]),
                                                      else_value=None, else_shape=shape_array([dynamic_dimension_value, 100]),
                                                      out_value=None, out_shape=shape_array([100, 100]),
                                                      auto_broadcast='none')
        self.assertTrue(flag, msg)

    def test_select_infer_tf_condition(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=None, condition_shape=shape_array([100]),
                                                      then_value=None, then_shape=shape_array([100, 20]),
                                                      else_value=None, else_shape=shape_array([100, 20]),
                                                      out_value=None, out_shape=shape_array([100, 20]),
                                                      auto_broadcast='numpy', fw_format='tf')
        self.assertTrue(flag, msg)

    def test_select_infer_tf_condition_dyn(self):
        flag, msg = self.build_select_graph_and_infer(condition_value=None,
                                                      condition_shape=shape_array([dynamic_dimension_value]),
                                                      then_value=None,
                                                      then_shape=shape_array([dynamic_dimension_value, 20]),
                                                      else_value=None,
                                                      else_shape=shape_array([dynamic_dimension_value, 20]),
                                                      out_value=None,
                                                      out_shape=shape_array([dynamic_dimension_value, 20]),
                                                      auto_broadcast='numpy', fw_format='tf')
        self.assertTrue(flag, msg)

    def test_select_infer_tf_condition_assert_raises(self):
        with self.assertRaisesRegex(AssertionError, "if 'condition' is a 1D tensor then it's size"):
            self.build_select_graph_and_infer(condition_value=None, condition_shape=shape_array([42]),
                                                      then_value=None, then_shape=shape_array([100, 20]),
                                                      else_value=None, else_shape=shape_array([100, 20]),
                                                      out_value=None, out_shape=shape_array([100, 20]),
                                                      auto_broadcast='numpy', fw_format='tf')

    def test_select_infer_assert_pdpd(self):
        with self.assertRaisesRegex(Error, "PDPD broadcasting rule is not implemented yet"):
            self.build_select_graph_and_infer(condition_value=None, condition_shape=[2, 2],
                                              then_value=None, then_shape=[2, 2],
                                              else_value=None, else_shape=[3, 3],
                                              out_value=None, out_shape=[42, 42],
                                              auto_broadcast='pdpd')

