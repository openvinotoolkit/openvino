# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.kaldi.add_reshape_transpose_around_conv_pool import AddReshapeTransposeAroundConvPool
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, connect_front, const, regular_op, shaped_parameter


class AddReshapeTransposeAroundConvPoolTests(unittest.TestCase):
    nodes = {
        **shaped_parameter('input', [1, 33]),
        **regular_op('some_op', {'op': 'some_op'}),
        **regular_op('splice', {'op': 'Splice', 'context': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]}),
        **regular_op('conv', {'kind': 'op', 'op': 'Convolution', 'kernel': [1, 11, 1, 5], 'patch_stride': 5,
                              'kernel_spatial': [1, 5]}),
        **regular_op('pool', {'kind': 'op', 'op': 'Pooling', 'pool_stride': 5, 'pool_step': [1, 1, 1, 1]}),
        **regular_op('out_op', {'op': "SomeOp"}),
    }

    ref_nodes = {
        **shaped_parameter('input', [1, 33]),
        **regular_op('some_op', {}),
        **regular_op('splice', {'op': 'Splice', 'context': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]}),

        **regular_op('shapeof', {'op': 'ShapeOf', 'type': 'ShapeOf'}),
        **const('ind', int64_array([0])),
        **const('axis', int64_array(0)),
        **regular_op('gather_batch', {'op': 'Gather', 'type': 'Gather'}),
        **const('t', int64_array([11])),
        **const('h', int64_array([5])),
        **const('ind_h', int64_array([1])),
        **regular_op('gather_h', {'op': "Gather", 'type': 'Gather'}),
        **const('th', int64_array([55])),
        **regular_op('div', {'op': 'Div', 'type': 'Divide'}),
        **regular_op('concat', {'op': 'Concat', 'type': 'Concat'}),

        **regular_op('reshape_in', {'op': 'Reshape', 'type': 'Reshape'}),
        **const('transpose_in_order', int64_array([0, 3, 1, 2])),
        **regular_op('transpose_in', {'op': 'Transpose', 'type': 'Transpose'}),
        **regular_op('conv', {'kind': 'op', 'op': 'Convolution', 'kernel': [1, 1, 11, 5]}),
        **regular_op('pool', {'kind': 'op', 'op': 'Pooling', 'pool_stride': 5, 'pool_step': [1, 1, 1, 1]}),
        **const('transpose_out_order', int64_array([0, 2, 3, 1])),
        **regular_op('transpose_out', {'op': 'Transpose', 'type': 'Transpose'}),
        **const('reshape_out_shape', int64_array([0, -1])),
        **regular_op('reshape_out', {'op': 'Reshape', 'type': 'Reshape'}),
        **regular_op('out_op', {'op': "SomeOp"})
    }

    def test_simple_convolution(self):
        graph = build_graph(self.nodes, [
            *connect_front('input', 'splice'),
            *connect_front('splice', 'conv'),
            *connect_front('conv', 'out_op')
        ], nodes_with_edges_only=True)
        graph.stage = 'front'
        AddReshapeTransposeAroundConvPool.find_and_replace_pattern(graph)

        ref_graph = build_graph(self.ref_nodes,
                                [
                                    *connect_front('input', 'splice'),
                                    *connect_front('splice', '0:reshape_in'),

                                    *connect_front('splice', 'shapeof'),
                                    *connect_front('shapeof:0', '0:gather_batch'),
                                    *connect_front('ind', '1:gather_batch'),
                                    *connect_front('axis', '2:gather_batch'),
                                    *connect_front('shapeof:0', '0:gather_h'),
                                    *connect_front('ind_h', '1:gather_h'),
                                    *connect_front('axis', '2:gather_h'),
                                    *connect_front('gather_h', '0:div'),
                                    *connect_front('th', '1:div'),
                                    *connect_front('gather_batch', '0:concat'),
                                    *connect_front('t', '1:concat'),
                                    *connect_front('h', '2:concat'),
                                    *connect_front('div', '3:concat'),
                                    *connect_front('concat', '1:reshape_in'),

                                    *connect_front('reshape_in', '0:transpose_in'),
                                    *connect_front('transpose_in_order', "1:transpose_in"),
                                    *connect_front('transpose_in', 'conv'),
                                    *connect_front('conv', '0:transpose_out'),
                                    *connect_front('transpose_out_order', '1:transpose_out'),
                                    *connect_front('transpose_out', '0:reshape_out'),
                                    *connect_front('reshape_out_shape', '1:reshape_out'),
                                    *connect_front('reshape_out', 'out_op')
                                ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_simple_convolution_wo_splice(self):
        graph = build_graph(self.nodes, [
            *connect_front('input', 'conv'),
            *connect_front('input', 'some_op'),
            *connect_front('conv', 'out_op')
        ], nodes_with_edges_only=True)
        graph.stage = 'front'
        AddReshapeTransposeAroundConvPool.find_and_replace_pattern(graph)

        ref_graph = build_graph(self.ref_nodes,
                                [
                                    *connect_front('input', '0:reshape_in'),
                                    *connect_front('input', 'some_op'),
                                    *connect_front('input', 'shapeof'),
                                    *connect_front('shapeof:0', '0:gather_batch'),
                                    *connect_front('ind', '1:gather_batch'),
                                    *connect_front('axis', '2:gather_batch'),
                                    *connect_front('shapeof:0', '0:gather_h'),
                                    *connect_front('ind_h', '1:gather_h'),
                                    *connect_front('axis', '2:gather_h'),
                                    *connect_front('gather_h', '0:div'),
                                    *connect_front('th', '1:div'),
                                    *connect_front('gather_batch', '0:concat'),
                                    *connect_front('t', '1:concat'),
                                    *connect_front('h', '2:concat'),
                                    *connect_front('div', '3:concat'),
                                    *connect_front('concat', '1:reshape_in'),

                                    *connect_front('reshape_in', '0:transpose_in'),
                                    *connect_front('transpose_in_order', "1:transpose_in"),
                                    *connect_front('transpose_in', 'conv'),
                                    *connect_front('conv', '0:transpose_out'),
                                    *connect_front('transpose_out_order', '1:transpose_out'),
                                    *connect_front('transpose_out', '0:reshape_out'),
                                    *connect_front('reshape_out_shape', '1:reshape_out'),
                                    *connect_front('reshape_out', 'out_op')
                                ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_op', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_simple_pooling(self):
        graph = build_graph(self.nodes, [
            *connect_front('input', 'splice'),
            *connect_front('splice', 'pool'),
            *connect_front('pool', 'out_op')
        ], nodes_with_edges_only=True)
        graph.stage = 'front'
        AddReshapeTransposeAroundConvPool.find_and_replace_pattern(graph)

        ref_graph = build_graph(self.ref_nodes,
                                [
                                    *connect_front('input', 'splice'),
                                    *connect_front('splice', '0:reshape_in'),

                                    *connect_front('splice', 'shapeof'),
                                    *connect_front('shapeof:0', '0:gather_batch'),
                                    *connect_front('ind', '1:gather_batch'),
                                    *connect_front('axis', '2:gather_batch'),
                                    *connect_front('shapeof:0', '0:gather_h'),
                                    *connect_front('ind_h', '1:gather_h'),
                                    *connect_front('axis', '2:gather_h'),
                                    *connect_front('gather_h', '0:div'),
                                    *connect_front('th', '1:div'),
                                    *connect_front('gather_batch', '0:concat'),
                                    *connect_front('t', '1:concat'),
                                    *connect_front('h', '3:concat'),
                                    *connect_front('div', '2:concat'),
                                    *connect_front('concat', '1:reshape_in'),

                                    *connect_front('reshape_in', '0:transpose_in'),
                                    *connect_front('transpose_in_order', "1:transpose_in"),
                                    *connect_front('transpose_in', 'pool'),
                                    *connect_front('pool', '0:transpose_out'),
                                    *connect_front('transpose_out_order', '1:transpose_out'),
                                    *connect_front('transpose_out', '0:reshape_out'),
                                    *connect_front('reshape_out_shape', '1:reshape_out'),
                                    *connect_front('reshape_out', 'out_op')
                                ])

        (flag, resp) = compare_graphs(graph, ref_graph, 'out_op', check_op_attrs=True)
        self.assertTrue(flag, resp)
