# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

import numpy as np
import pytest

from openvino.tools.mo.back.compress_quantized_weights import CompressQuantizeWeights, ZeroPointOptimizer
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.elementwise import Sub, Mul
from openvino.tools.mo.ops.fakequantize import FakeQuantize
from openvino.tools.mo.front.common.partial_infer.eltwise import eltwise_infer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from openvino.tools.mo.middle.passes.infer import type_infer
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, valued_const_with_data, result, connect, \
    shaped_const_with_data


def nodes_dict(original, transformed=None, levels=255, data=None,
               il=[-127], ih=[127], ol=[-127], oh=[127],
               scale=np.array([1]), zp=np.array([0]), int_data=None):
    shape = [1, 2, 3, 4] if data is None else np.array(data).shape
    data = np.ones(shape, dtype=original) if data is None else np.array(data, dtype=original)
    if int_data is None:
        int_data = data.astype(dtype=np.int8)
    transformed = transformed if transformed is not None else original

    return {
        **valued_const_with_data('weights', data),
        **valued_const_with_data('int_weights', int_data),

        **regular_op_with_shaped_data(
            'weights_cast', shape, {'type': 'Convert', 'op': 'Cast', 'infer': Cast.infer, 'dst_type': np.float32}),

        **regular_op_with_shaped_data(
            'cast', shape, {'type': 'Convert', 'op': 'Cast', 'infer': Cast.infer, 'dst_type': transformed}),

        **valued_const_with_data('il', np.array(il)),
        **valued_const_with_data('ih', np.array(ih)),
        **valued_const_with_data('ol', np.array(ol)),
        **valued_const_with_data('oh', np.array(oh)),

        **regular_op_with_shaped_data(
            'FQ', shape, {'type': 'FakeQuantize', 'infer': FakeQuantize.infer, 'stop_value_propagation': True,
                          'levels': levels, 'op': 'FakeQuantize'}),

        **valued_const_with_data('zp', zp),
        **valued_const_with_data('scale', scale),

        **regular_op_with_shaped_data(
            'sub', shape, {'type': 'Subtract', 'op': 'Sub', 'infer': lambda node: eltwise_infer(node, Sub.operation)}),

        **regular_op_with_shaped_data(
            'mul', shape, {'type': 'Multiply', 'op': 'Mul', 'infer': lambda node: eltwise_infer(node, Mul.operation)}),

        **result()
    }


class CompressionQuantizeDequantizeSeparateTest(unittest.TestCase):
    def test_quantize(self):
        original_type = np.float32
        nodes = nodes_dict(original_type)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        error_message = 'Unexpected number of FakeQuantize nodes {} CompressQuantizeWeights.quantize_data call `{}`'
        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 1, error_message.format('before', len(fq_nodes)))
        fake_quantize = fq_nodes[0]

        CompressQuantizeWeights.quantize_data(fake_quantize, original_type, np.int8, "signed")
        graph.clean_up()

        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 1, error_message.format('after', len(fq_nodes)))
        self.assertEqual(fq_nodes[0].in_port(0).get_source().node.soft_get('type'), 'Const')
        self.assertEqual(fq_nodes[0].in_port(0).get_source().node.data_type, np.int8)

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_dequantize(self):
        original_type = np.float32
        nodes = nodes_dict(original_type, np.int8)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:cast'),
            *connect('cast:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        error_message = 'Unexpected number of {} nodes {} CompressQuantizeWeights.dequantize_data call `{}`'
        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        cast_nodes = graph.get_op_nodes(name='cast')
        self.assertEqual(len(fq_nodes), 1, error_message.format('FakeQuantize', 'before', len(fq_nodes)))
        self.assertEqual(len(cast_nodes), 1, error_message.format('Convert', 'before', len(cast_nodes)))
        cast_nodes[0]['need_shape_inference'] = True

        CompressQuantizeWeights.dequantize_data(fq_nodes[0], original_type, np.int8)
        graph.clean_up()

        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 0, error_message.format('FakeQuantize', 'after', len(fq_nodes)))

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:cast'),
            *connect('cast:0', '0:sub'),
            *connect('zp:0', '1:sub'),
            *connect('sub:0', '0:mul'),
            *connect('scale:0', '1:mul'),
            *connect('mul:0', 'output'),
        ], {'cast': {'dst_type': original_type}}, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_quantize_new_fp16(self):
        original_type = np.float16
        nodes = nodes_dict(original_type)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        error_message = 'Unexpected number of FakeQuantize nodes {} CompressQuantizeWeights.quantize_data call `{}`'
        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 1, error_message.format('before', len(fq_nodes)))
        fake_quantize = fq_nodes[0]

        CompressQuantizeWeights.quantize_data(fake_quantize, original_type, np.int8, "signed")
        graph.clean_up()

        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        self.assertEqual(len(fq_nodes), 1, error_message.format('after', len(fq_nodes)))
        self.assertEqual(fq_nodes[0].in_port(0).get_source().node.soft_get('type'), 'Const')
        self.assertEqual(fq_nodes[0].in_port(0).get_source().node.data_type, np.int8)

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)


class TestCompressionDataTypeTest():
    @pytest.mark.parametrize("original",[np.int64,
                np.int32,
                np.float64,
                np.float32,
                np.float16])
    def test_data_type(self, original):
        nodes = nodes_dict(original)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True, cli=Namespace(static_shape=True))

        CompressQuantizeWeights().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:cast'),
            *connect('cast:0', '0:sub'),
            *connect('zp:0', '1:sub'),
            *connect('sub:0', '0:mul'),
            *connect('scale:0', '1:mul'),
            *connect('mul:0', 'output'),
        ], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

    def test_data_type_new_fp16(self):
        nodes = nodes_dict(np.float16)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:weights_cast'),
            *connect('weights_cast:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True, cli=Namespace(data_type='FP16', static_shape=True))

        CompressQuantizeWeights().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:weights_cast'),
            *connect('weights_cast:0', '0:sub'),
            *connect('zp:0', '1:sub'),
            *connect('sub:0', '0:mul'),
            *connect('scale:0', '1:mul'),
            *connect('mul:0', 'output'),
        ], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp


    def test_fp16_fake_quantize(self):
        original_type = np.float16
        input_low = np.array([-0.59033203125, -1.4833984375, -1.2900390625], dtype=np.float16)
        input_high = np.array([0.59033203125, 1.4833984375, 1.2900390625], dtype=np.float16)
        output_low = np.array([0.295166015625, 0.74169921875, 0.64501953125], dtype=np.float16)
        output_high = np.array([-0.295166015625, -0.74169921875, -0.64501953125], dtype=np.float16)
        scale = np.array([-0.002325, -0.00584, -0.005077], dtype=np.float16)
        int_data = np.array([43, 103, 118], dtype=np.int8)
        nodes = nodes_dict(original_type, transformed=np.int8,
                           levels=255, data=np.array([0.2, 1.2, 1.2], dtype=np.float16),
                           il=input_low, ih=input_high, ol=output_low, oh=output_high, scale=scale, int_data=int_data)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)
        type_infer(graph)

        error_message = 'Unexpected number of {} nodes {} CompressQuantizeWeights.dequantize_data call `{}`'
        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        assert len(fq_nodes) == 1, error_message.format('FakeQuantize', 'before', len(fq_nodes))

        CompressQuantizeWeights().find_and_replace_pattern(graph)
        graph.clean_up()
        ZeroPointOptimizer().find_and_replace_pattern(graph)
        graph.clean_up()

        fq_nodes = graph.get_op_nodes(type='FakeQuantize')
        assert len(fq_nodes) == 0, error_message.format('FakeQuantize', 'after', len(fq_nodes))

        graph_ref = build_graph(nodes, [
            *connect('int_weights:0', '0:cast'),
            *connect('cast:0', '0:mul'),
            *connect('scale:0', '1:mul'),
            *connect('mul:0', 'output'),
        ], {'cast': {'dst_type': original_type}}, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp


class TestAccuracyCheckFP32Test():
    eps = np.finfo(np.float32).eps

    @pytest.mark.parametrize("data, in_low, in_high, out_low, out_high, levels, add_cast" ,[
        ([-2.586, -1.338, 2.773, 4.414], [-2.586], [4.414], [-2.586], [4.414], 256, False),
        ([-1.5, -0.32, 0.167, 2.8], [-1.5], [2.8], [-1.5], [2.8], 256, False),
        ([1, 1 + eps, 1 + 2 * eps, 1 + 3 * eps], [1], [1 + 3 * eps], [1], [1 + 3 * eps], 256, False),
        ([1.0, 2.0, 3.0, 4.0], [1], [4], [1], [4], 256, False),
        ([-2.586, -1.338, 2.773, 4.414], [-2.586], [4.414], [-2.586], [4.414], 256, True),
        ([-1.5, -0.32, 0.167, 2.8], [-1.5], [2.8], [-1.5], [2.8], 256, True),
        ([1, 1 + eps, 1 + 2 * eps, 1 + 3 * eps], [1], [1 + 3 * eps], [1], [1 + 3 * eps], 256, True),
        ([1.0, 2.0, 3.0, 4.0], [1], [4], [1], [4], 256, True),
    ])
    def test_accuracy(self, data, in_low, in_high, out_low, out_high, levels, add_cast):
        if not add_cast:
            nodes = nodes_dict(np.float32, None, levels, data, in_low, in_high, out_low, out_high)
            graph = build_graph(nodes, [
                *connect('weights:0', '0:FQ'),
                *connect('il:0', '1:FQ'),
                *connect('ih:0', '2:FQ'),
                *connect('ol:0', '3:FQ'),
                *connect('oh:0', '4:FQ'),
                *connect('FQ:0', 'output'),
            ], nodes_with_edges_only=True)
        else:
            nodes = nodes_dict(np.float16, None, levels, data, in_low, in_high, out_low, out_high)
            graph = build_graph(nodes, [
                *connect('weights:0', '0:weights_cast'),
                *connect('weights_cast:0', '0:FQ'),
                *connect('il:0', '1:FQ'),
                *connect('ih:0', '2:FQ'),
                *connect('ol:0', '3:FQ'),
                *connect('oh:0', '4:FQ'),
                *connect('FQ:0', 'output'),
            ], nodes_with_edges_only=True)
        graph_ref = graph.copy()

        CompressQuantizeWeights().find_and_replace_pattern(graph)

        for node in graph.get_op_nodes() + graph_ref.get_op_nodes():
            node['stop_value_propagation'] = False
            node['need_shape_inference'] = node.soft_get('need_shape_inference', True)

        graph.clean_up()
        graph_ref.clean_up()

        const_result_graph = build_graph({**shaped_const_with_data('weights', np.array(data).shape), **result()},
                                         [*connect('weights', 'output')], nodes_with_edges_only=True)
        (flag, resp) = compare_graphs(graph, const_result_graph, 'output', check_op_attrs=True)
        assert flag, resp

        (flag, resp) = compare_graphs(graph_ref, const_result_graph, 'output', check_op_attrs=True)
        assert flag, resp

        # as this two graphs calculated the same data through different constant folding functions, they resulted in
        # constants of different data type since FakeQuantize always have f32 output dtype, but eltwises use numpy
        # for folding which doesn't have such restriction
        const_node = graph.get_op_nodes(type='Const')
        assert len(const_node) == 1
        if const_node[0].data_type == np.float64:
            const_node[0].data_type = np.float32

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

        # I would like to leave this commented code here to quickly check the actual output value:
        # print(result_node.in_port(0).data.get_value())  # actual calculated value


class TestNegativeCompressionTestLevels():
    @pytest.mark.parametrize("levels" , [(2), (257), (None), (0), (-5)])
    def test_negative_fq_unacceptable_levels(self, levels):
        nodes = nodes_dict(np.float32, None, levels)

        graph = build_graph(nodes, [
            *connect('weights:0', '0:FQ'),
            *connect('il:0', '1:FQ'),
            *connect('ih:0', '2:FQ'),
            *connect('ol:0', '3:FQ'),
            *connect('oh:0', '4:FQ'),
            *connect('FQ:0', 'output'),
        ], nodes_with_edges_only=True)
        graph_ref = graph.copy()
        CompressQuantizeWeights().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp



class TestZeroPointOptimizerTestClass():
    @pytest.mark.parametrize("weights, zero_point, adj_weights, adj_zero_point" ,[
        ([-10, 7], [-1], [-9, 8], [0]),
        ([-10, 7], [-0.99999999], [-9, 8], [0]),
    ])
    def test_zero_point_optimization(self, weights, zero_point, adj_weights, adj_zero_point):
        nodes = lambda w, zp: {
            **valued_const_with_data('weights', np.array(w, dtype=np.int8)),
            **regular_op_with_shaped_data(
                'cast', [len(w)], {'type': 'Convert', 'op': 'Cast', 'infer': Cast.infer, 'dst_type': np.float32}),
            **valued_const_with_data('zp', np.array(zp, dtype=np.float32)),
            **regular_op_with_shaped_data(
                'sub', [len(w)],
                {'type': 'Subtract', 'op': 'Sub', 'infer': lambda node: eltwise_infer(node, Sub.operation)}),
            **result()
        }
        edges = [
            *connect("weights:0", "0:cast"),
            *connect("cast:0", "0:sub"),
            *connect("zp:0", "1:sub"),
            *connect("sub:0", "0:output"),
        ]
        graph = build_graph(nodes(weights, zero_point), edges, nodes_with_edges_only=True)
        ZeroPointOptimizer().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes(adj_weights, adj_zero_point), [
            *connect("weights:0", "0:cast"),
            *connect("cast:0", "0:output"),
        ], nodes_with_edges_only=True)
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp

    @pytest.mark.parametrize("weights, zero_point, adj_weights, adj_zero_point" ,[
        ([-128, 7], [1], [-128, 7], [1]),
        ([127, 7], [-1], [127, 7], [-1]),
    ])
    def test_negative_zero_point_optimization(self, weights, zero_point, adj_weights, adj_zero_point):
        nodes = lambda w, zp: {
            **valued_const_with_data('weights', np.array(w, dtype=np.int8)),
            **regular_op_with_shaped_data(
                'cast', [len(w)], {'type': 'Convert', 'op': 'Cast', 'infer': Cast.infer, 'dst_type': np.float32}),
            **valued_const_with_data('zp', np.array(zp, dtype=np.float32)),
            **regular_op_with_shaped_data(
                'sub', [len(w)],
                {'type': 'Subtract', 'op': 'Sub', 'infer': lambda node: eltwise_infer(node, Sub.operation)}),
            **result()
        }
        edges = [
            *connect("weights:0", "0:cast"),
            *connect("cast:0", "0:sub"),
            *connect("zp:0", "1:sub"),
            *connect("sub:0", "0:output"),
        ]
        graph = build_graph(nodes(weights, zero_point), edges, nodes_with_edges_only=True)
        ZeroPointOptimizer().find_and_replace_pattern(graph)
        graph.clean_up()

        graph_ref = build_graph(nodes(adj_weights, adj_zero_point), edges, nodes_with_edges_only=True)
        graph_ref.clean_up()

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp
