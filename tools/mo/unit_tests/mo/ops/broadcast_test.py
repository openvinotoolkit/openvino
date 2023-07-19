# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, undefined_shape_of_rank
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.broadcast import Broadcast
from unit_tests.utils.graph import build_graph, valued_const_with_data, regular_op_with_empty_data, \
    shaped_data


class BroadcastTest(unittest.TestCase):
    #test_broadcast
    def run_test_case(self, data, target_shape, axes_mapping=None, mode='numpy', ref_out=None, test_raising=False):
        if ref_out is not None:
            input_data = valued_const_with_data('data', int64_array(data))
        else:
            input_data = shaped_data('data', int64_array(data))

        nodes = {
            **input_data,
            **valued_const_with_data('target_shape', int64_array(target_shape)),
            **regular_op_with_empty_data('broadcast', {'op': 'Broadcast', 'mode': mode}),
        }

        edges = [('data', 'broadcast'),
                 ('target_shape', 'broadcast'),
                 ('broadcast', 'broadcast_d')]

        if axes_mapping is not None:
            nodes.update(**valued_const_with_data('axes_mapping', int64_array(axes_mapping)))
            edges.append(('axes_mapping', 'broadcast'))
        graph = build_graph(nodes, edges)

        broadcast_node = Node(graph, 'broadcast')
        if test_raising:
            self.assertRaises(AssertionError, Broadcast.infer, broadcast_node)
            return

        Broadcast.infer(broadcast_node)
        if ref_out is not None:
            self.assertTrue(np.array_equal(broadcast_node.out_node().value, np.array(ref_out)))
        else:
            self.assertTrue(np.array_equal(broadcast_node.out_node().shape, np.array(target_shape)))

    def test_case_1(self):
        data = [1]
        target_shape = [3, 3]
        mode = 'numpy'
        ref_out = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        self.run_test_case(data, target_shape, mode=mode, ref_out=ref_out)

    def test_case_2(self):
        data = [1]
        target_shape = [3, 3]
        mode = 'numpy'
        self.run_test_case(data, target_shape, mode=mode)

# shape broadcasting
    def test_case_3(self):
        data = [1]
        target_shape = [1, 2]
        axes_mapping = [0]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode)

    def test_case_4(self):
        data = [1]
        target_shape = [1, 2]
        axes_mapping = [-2]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode)

    def test_case_5(self):
        data = [1, 7]
        target_shape = [5, 1, 7, 3]
        axes_mapping = [1, 2]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode)

    def test_case_6(self):
        data = [2, 1, 3]
        target_shape = [2, 1, 3, 3]
        axes_mapping = [0, 1, 2]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode)

    def test_case_7(self):
        data = [2, 1, 3]
        target_shape = [5, 2, 1, 3]
        axes_mapping = [1, 2, 3]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode)

# value broadcasting
    def test_case_8(self):
        data = [1]
        target_shape = [1, 2]
        axes_mapping = [0]
        mode = 'explicit'
        ref_out = [[1, 1]]
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, ref_out=ref_out)

    def test_case_9(self):
        data = [[3, 1]]
        target_shape = [2, 1, 2]
        axes_mapping = [1, 2]
        mode = 'explicit'
        ref_out = [[[3, 1]], [[3, 1]]]  # ref_shape (2, 1, 2)
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, ref_out=ref_out)

    def test_case_10(self):
        data = [[3, 1]]
        target_shape = [2, 1, 2]
        axes_mapping = [-2, -1]
        mode = 'explicit'
        ref_out = [[[3, 1]], [[3, 1]]]  # ref_shape (2, 1, 2)
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, ref_out=ref_out)

    def test_case_11(self):
        data = [[[9, 5, 7]], [[9, 5, 7]]]
        target_shape = [2, 2, 1, 3]
        axes_mapping = [1, 2, 3]
        mode = 'explicit'
        ref_out = [[[[9, 5, 7]], [[9, 5, 7]]], [[[9, 5, 7]], [[9, 5, 7]]]]  # in_shape (2, 1, 3)
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, ref_out=ref_out)

    def test_case_12(self):
        data = [[[9, 5, 7]], [[3, 4, 8]]]
        target_shape = [2, 1, 3, 3]
        axes_mapping = [0, 1, 2]
        mode = 'explicit'
        ref_out = [[[[9, 9, 9], [5, 5, 5], [7, 7, 7]]], [[[3, 3, 3], [4, 4, 4], [8, 8, 8]]]]  # in_shape (2, 1, 3)
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, ref_out=ref_out)

# negative tests
    def test_case_13(self):
        data = [1]
        target_shape = [2, 2]
        axes_mapping = [0]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, test_raising=True)

    def test_case_14(self):
        data = [1, 7]
        target_shape = [5, 2, 7, 3]
        axes_mapping = [1, 2]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, test_raising=True)

    def test_case_15(self):
        data = [1, 7]
        target_shape = [5, 2, 7, 3]
        axes_mapping = [2, 1]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, test_raising=True)

    def test_case_16(self):
        data = [1, 7]
        target_shape = [5, 2, 7, 3]
        axes_mapping = [-3, -2]
        mode = 'explicit'
        self.run_test_case(data, target_shape, axes_mapping=axes_mapping, mode=mode, test_raising=True)


#test_broadcast_dynamic
    def test_broadcast_case_1(self):
        data = [1]
        target_shape_shape = [3]
        axes_mapping = [0]
        mode = 'explicit'
        ref_out_shape = undefined_shape_of_rank(3)
        self._test_broadcast(data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=False)

    def test_broadcast_case_2(self):
        data = [1]
        target_shape_shape = [3]
        axes_mapping = None
        mode = 'numpy'
        ref_out_shape = undefined_shape_of_rank(3)
        self._test_broadcast(data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=False)

    def test_broadcast_case_3(self):
        data = [1]
        target_shape_shape = [3]
        axes_mapping = None
        mode = 'bidirectional'
        ref_out_shape = undefined_shape_of_rank(3)
        self._test_broadcast(data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=False)

    def test_broadcast_case_4(self):
        data = [1, 7]
        target_shape_shape = [4]
        axes_mapping = [1, 2]
        mode = 'explicit'
        ref_out_shape = undefined_shape_of_rank(4)
        self._test_broadcast(data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=False)

    def test_broadcast_case_5(self):
        data = [1, 2]
        target_shape_shape = [3]
        axes_mapping = None
        mode = 'numpy'
        ref_out_shape = undefined_shape_of_rank(3)
        self._test_broadcast(data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=False)

    def test_broadcast_case_6(self):
        data = [1, 1]
        target_shape_shape = [2]
        axes_mapping = None
        mode = 'bidirectional'
        ref_out_shape = undefined_shape_of_rank(2)
        self._test_broadcast(data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=False)

    def test_broadcast_case_7(self):
        data = [1, 1]
        target_shape_shape = [2, 1]
        axes_mapping = None
        mode = 'numpy'
        ref_out_shape = None
        test_raising = True
        self._test_broadcast(data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=test_raising)

    def _test_broadcast(self, data, target_shape_shape, axes_mapping, mode, ref_out_shape, test_raising=False):
        nodes = {
            **shaped_data('data', int64_array(data)),
            **shaped_data('target_shape', int64_array(target_shape_shape)),
            **regular_op_with_empty_data('broadcast', {'op': 'Broadcast', 'mode': mode}),
        }

        edges = [('data', 'broadcast'),
                 ('target_shape', 'broadcast'),
                 ('broadcast', 'broadcast_d')]

        if axes_mapping is not None:
            nodes.update(**valued_const_with_data('axes_mapping', int64_array(axes_mapping)))
            edges.append(('axes_mapping', 'axes_mapping_d'))
            edges.append(('axes_mapping_d', 'broadcast'))
        graph = build_graph(nodes, edges)

        broadcast_node = Node(graph, 'broadcast')
        if test_raising:
            with self.assertRaises(AssertionError):
                Broadcast.infer(broadcast_node)
        else:
            Broadcast.infer(broadcast_node)
            self.assertTrue(np.array_equal(broadcast_node.out_node().shape, ref_out_shape))
