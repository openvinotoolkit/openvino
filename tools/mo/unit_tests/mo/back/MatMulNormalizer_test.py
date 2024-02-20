# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

import numpy as np
import pytest

from openvino.tools.mo.back.MatMulNormalizer import SmartReshape_HC_Reshape_MatMul, PullTransposeThroughFQUp
from openvino.tools.mo.ops.MatMul import MatMul
from openvino.tools.mo.ops.fakequantize import FakeQuantize
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, \
    shaped_const_with_data, result, connect, connect_data
from unit_tests.utils.graph import regular_op_with_empty_data as op_with_empty_data


class TestSmartReshape_HC_Reshape_MatMulTest():
    @pytest.mark.parametrize("in1_shape, in2_shape, reshape_pattern, transpose_a, transpose_b, updated_pattern",
        [
            ([1, 20, 30], [30, 40], [20, -1], False, False, [-1, 30]),
            ([1, 20, 30], [40, 30], [20, -1], False, True, [-1, 30]),
            ([1, 30, 20], [30, 40], [-1, 20], True, False, [30, -1]),
            ([1, 30, 20], [40, 30], [-1, 20], True, True, [30, -1]),
        ]
    )
    def test_reshape_on_the_A_input(self,
                                    in1_shape, in2_shape, reshape_pattern, transpose_a, transpose_b, updated_pattern):
        nodes = {
            **regular_op_with_shaped_data('in_1', in1_shape, dict(type='Parameter', op='Parameter')),
            **regular_op_with_shaped_data('in_2', in2_shape, dict(type='Parameter', op='Parameter')),
            **valued_const_with_data('dim', int64_array(reshape_pattern)),
            **op_with_empty_data('reshape',
                                 dict(type='Reshape', op='Reshape', infer=Reshape.infer, need_shape_inference=True)),
            **op_with_empty_data('matmul',
                                 dict(type='MatMul', op='MatMul', infer=MatMul.infer, need_shape_inference=True,
                                      transpose_a=transpose_a, transpose_b=transpose_b, dim_attrs={})),
            **result(),
        }
        edges = [
            *connect('in_1:0', '0:reshape'),
            *connect('dim:0', '1:reshape'),
            *connect('reshape:0', '0:matmul'),
            *connect('in_2:0', '1:matmul'),
            *connect('matmul:0', 'output'),
        ]
        graph = build_graph(nodes_attrs=nodes, edges=edges, cli=Namespace(static_shape=True))
        graph.clean_up()
        SmartReshape_HC_Reshape_MatMul().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes_attrs=nodes, edges=edges, update_attributes={
            'dim': {'value': int64_array(updated_pattern)}, 'dim_d': {'value': int64_array(updated_pattern)}})
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

    @pytest.mark.parametrize("in1_shape, in2_shape, reshape_pattern, transpose_a, transpose_b, updated_pattern",[
        ([20, 30], [1, 30, 40], [-1, 40], False, False, [30, -1]),
        ([20, 30], [1, 40, 30], [40, -1], False, True, [-1, 30]),
        ([30, 20], [1, 30, 40], [-1, 40], True, False, [30, -1]),
        ([30, 20], [1, 40, 30], [40, -1], True, True, [-1, 30]),
    ])
    def test_reshape_on_the_B_input(self,
                                    in1_shape, in2_shape, reshape_pattern, transpose_a, transpose_b, updated_pattern):
        nodes = {
            **regular_op_with_shaped_data('in_1', in1_shape, dict(type='Parameter', op='Parameter')),
            **regular_op_with_shaped_data('in_2', in2_shape, dict(type='Parameter', op='Parameter')),
            **valued_const_with_data('dim', int64_array(reshape_pattern)),
            **op_with_empty_data('reshape',
                                 dict(type='Reshape', op='Reshape', infer=Reshape.infer, need_shape_inference=True)),
            **op_with_empty_data('matmul',
                                 dict(type='MatMul', op='MatMul', infer=MatMul.infer, need_shape_inference=True,
                                      transpose_a=transpose_a, transpose_b=transpose_b, dim_attrs={})),
            **result(),
        }
        edges = [
            *connect('in_1:0', '0:matmul'),
            *connect('in_2:0', '0:reshape'),
            *connect('dim:0', '1:reshape'),
            *connect('reshape:0', '1:matmul'),
            *connect('matmul:0', 'output'),
        ]
        graph = build_graph(nodes_attrs=nodes, edges=edges, cli=Namespace(static_shape=True))
        graph.clean_up()
        SmartReshape_HC_Reshape_MatMul().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes_attrs=nodes, edges=edges, update_attributes={
            'dim': {'value': int64_array(updated_pattern)}, 'dim_d': {'value': int64_array(updated_pattern)}})
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp


class FQTransposePullerTest(unittest.TestCase):
    def nodes(self, input_shape, transpose_shape, fq_shape, is_input_const):
        nodes = {
            **valued_const_with_data('il', np.array([[[[0]]]])),
            **valued_const_with_data('ih', np.array([[[[255]]]])),
            **valued_const_with_data('ol', np.array([[[[0]]]])),
            **valued_const_with_data('oh', np.array([[[[255]]]])),
            **regular_op_with_shaped_data('FQ', fq_shape, dict(type='FakeQuantize', op='FakeQuantize', infer=FakeQuantize.infer)),
            **valued_const_with_data('order', int64_array([0, 2, 3, 1])),
            **regular_op_with_shaped_data('transpose', transpose_shape, dict(type='Transpose', op='Transpose', infer=Transpose.infer)),
            **regular_op_with_shaped_data('relu', fq_shape, dict(type='Relu', op='Relu')),

            **result(),
        }

        if is_input_const:
            input_node = shaped_const_with_data('input', input_shape)
        else:
            input_node = regular_op_with_shaped_data('input', input_shape, dict(type='Parameter', op='Parameter'))

        nodes.update(input_node)
        return nodes

    def test_positive(self):
        nodes = self.nodes([1, 3, 224, 224], [1, 224, 224, 3], [1, 3, 224, 224], True)
        edges = [
            *connect('input', '0:FQ'),
            *connect('il', '1:FQ'),
            *connect('ih', '2:FQ'),
            *connect('ol', '3:FQ'),
            *connect('oh', '4:FQ'),
            *connect('FQ:0', '0:transpose'),
            *connect('order:0', '1:transpose'),
            *connect('transpose:0', 'output'),
        ]
        graph = build_graph(nodes_attrs=nodes, edges=edges, nodes_with_edges_only=True)
        PullTransposeThroughFQUp().find_and_replace_pattern(graph)
        graph.clean_up()

        nodes = self.nodes([1, 3, 224, 224], [1, 224, 224, 3], [1, 224, 224, 3], True)
        edges = [
            *connect('input', '0:transpose'),
            *connect('order:0', '1:transpose'),
            *connect('transpose', '0:FQ'),
            *connect('il', '1:FQ'),
            *connect('ih', '2:FQ'),
            *connect('ol', '3:FQ'),
            *connect('oh', '4:FQ'),
            *connect('FQ:0', 'output'),
        ]
        graph_ref = build_graph(nodes_attrs=nodes, edges=edges, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative_1(self):
        nodes = self.nodes([1, 3, 224, 224], [1, 224, 224, 3], [1, 3, 224, 224], True)
        edges = [
            *connect('input', '0:FQ'),
            *connect('il', '1:FQ'),
            *connect('ih', '2:FQ'),
            *connect('ol', '3:FQ'),
            *connect('oh', '4:FQ'),
            *connect('FQ:0', '0:transpose'),
            *connect_data('FQ:0', 'relu'),
            *connect('order:0', '1:transpose'),
            *connect('transpose:0', 'output'),
        ]
        graph = build_graph(nodes_attrs=nodes, edges=edges, nodes_with_edges_only=True)
        graph_ref = graph.copy()
        PullTransposeThroughFQUp().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_negative_2(self):
        nodes = self.nodes([1, 3, 224, 224], [1, 224, 224, 3], [1, 3, 224, 224], False)
        edges = [
            *connect('input', '0:FQ'),
            *connect('il', '1:FQ'),
            *connect('ih', '2:FQ'),
            *connect('ol', '3:FQ'),
            *connect('oh', '4:FQ'),
            *connect('FQ:0', '0:transpose'),
            *connect('order:0', '1:transpose'),
            *connect('transpose:0', 'output'),
        ]
        graph = build_graph(nodes_attrs=nodes, edges=edges, nodes_with_edges_only=True)
        graph_ref = graph.copy()
        PullTransposeThroughFQUp().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
