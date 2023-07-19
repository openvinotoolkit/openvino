# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.middle.passes.convert_data_type import packed_U4, packed_I4
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import valued_const_with_data, regular_op_with_empty_data, result, build_graph, connect

nodes = lambda value, dst_type: {
    **valued_const_with_data('value', np.array(value)),
    **regular_op_with_empty_data('convert', {'dst_type': dst_type, 'infer': Cast.infer}),
    **result(),
}


class CastTest(unittest.TestCase):
    """
    Example of checking:
        7 == 0111,           padded to 0111 0000, results in 112
        7 == 0111, 8 == 1000 packed to 0111 1000, results in 120

        -8 == 1000,          padded to 1000 0000, results in 128
    """

    def test_case_1(self):
        value = [0]
        expected = [0]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_2(self):
        value = [1]
        expected = [16]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_3(self):
        value = [2]
        expected = [32]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_4(self):
        value = [3]
        expected = [48]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_5(self):
        value = [4]
        expected = [64]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_6(self):
        value = [5]
        expected = [80]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_7(self):
        value = [6]
        expected = [96]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_8(self):
        value = [7]
        expected = [112]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_9(self):
        value = [8]
        expected = [128]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_10(self):
        value = [9]
        expected = [144]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_11(self):
        value = [10]
        expected = [160]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_12(self):
        value = [11]
        expected = [176]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_13(self):
        value = [12]
        expected = [192]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_14(self):
        value = [13]
        expected = [208]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_15(self):
        value = [14]
        expected = [224]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_16(self):
        value = [15]
        expected = [240]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_17(self):
        value = [0, 15]
        expected = [15]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_18(self):
        value = [1, 14]
        expected = [30]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_19(self):
        value = [2, 13]
        expected = [45]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_20(self):
        value = [3, 12]
        expected = [60]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_21(self):
        value = [4, 11]
        expected = [75]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_22(self):
        value = [5, 10]
        expected = [90]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_23(self):
        value = [6, 9]
        expected = [105]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_24(self):
        value = [7, 8]
        expected = [120]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_25(self):
        value = [8, 7]
        expected = [135]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_26(self):
        value = [9, 6]
        expected = [150]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_27(self):
        value = [10, 5]
        expected = [165]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_28(self):
        value = [11, 4]
        expected = [180]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_29(self):
        value = [12, 3]
        expected = [195]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_30(self):
        value = [13, 2]
        expected = [210]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_31(self):
        value = [14, 1]
        expected = [225]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_32(self):
        value = [15, 0]
        expected = [240]
        custom_dtype = packed_U4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_33(self):
        value = [-8]
        expected = [128]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_34(self):
        value = [-7]
        expected = [144]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_35(self):
        value = [-6]
        expected = [160]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_36(self):
        value = [-5]
        expected = [176]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_37(self):
        value = [-4]
        expected = [192]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_38(self):
        value = [-3]
        expected = [208]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_39(self):
        value = [-2]
        expected = [224]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_40(self):
        value = [-1]
        expected = [240]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_41(self):
        value = [0]
        expected = [0]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_42(self):
        value = [1]
        expected = [16]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_43(self):
        value = [2]
        expected = [32]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_44(self):
        value = [3]
        expected = [48]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_45(self):
        value = [4]
        expected = [64]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_46(self):
        value = [5]
        expected = [80]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_47(self):
        value = [6]
        expected = [96]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_48(self):
        value = [7]
        expected = [112]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_49(self):
        value = [-8, 7]
        expected = [135]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_50(self):
        value = [-7, 6]
        expected = [150]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_51(self):
        value = [-6, 5]
        expected = [165]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_52(self):
        value = [-5, 4]
        expected = [180]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_53(self):
        value = [-4, 3]
        expected = [195]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_54(self):
        value = [-3, 2]
        expected = [210]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_55(self):
        value = [-2, 1]
        expected = [225]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_56(self):
        value = [-1, 0]
        expected = [240]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_57(self):
        value = [0, -1]
        expected = [15]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_58(self):
        value = [1, -2]
        expected = [30]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_59(self):
        value = [2, -3]
        expected = [45]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_60(self):
        value = [3, -4]
        expected = [60]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_61(self):
        value = [4, -5]
        expected = [75]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_62(self):
        value = [5, -6]
        expected = [90]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_63(self):
        value = [6, -7]
        expected = [105]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)

    def test_case_64(self):
        value = [7, -8]
        expected = [120]
        custom_dtype = packed_I4
        self._test_custom_value_propagation(value, expected, custom_dtype)
    # Add more test cases as needed


    def _test_custom_value_propagation(self, value, expected, custom_dtype):
        graph = build_graph(nodes(value, custom_dtype), [
            *connect('value', 'convert'), *connect('convert', 'output'),
        ])
        partial_infer(graph)

        graph_ref = build_graph(nodes(value, custom_dtype), [
            *connect('value', 'convert'), *connect('convert', 'output')],
                                {'convert_d': {'force_type': custom_dtype, 'force_shape': np.array(value).shape,
                                               'value': expected}})

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
