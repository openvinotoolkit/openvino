# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, float32_array, int8_array
from openvino.tools.mo.middle.quantize_dequantize_linear_resolver import QuantizeDequantizeLinearResolver
from openvino.tools.mo.middle.quantize_linear_resolver import QuantizeLinearResolver
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, result, connect, connect_data, \
    valued_const_with_data, regular_op_with_empty_data

nodes_attributes = {
    **valued_const_with_data('y_scale_1', float32_array(1.0 / 255.0)),
    **valued_const_with_data('y_scale_2', float32_array(1.0 / 255.0)),
    **valued_const_with_data('y_zeropoint_1', int8_array(0)),
    **valued_const_with_data('y_zeropoint_2', int8_array(0)),
    **valued_const_with_data('x_scale_1', float32_array(1.0 / 255.0)),
    **valued_const_with_data('x_scale_2', float32_array(1.0 / 255.0)),
    **valued_const_with_data('x_zeropoint_1', int8_array(0)),
    **valued_const_with_data('x_zeropoint_2', int8_array(0)),
    **valued_const_with_data('const_input', float32_array([[0.3, 0.6], [-0.7, -0.9]])),
    **valued_const_with_data('in_low', float32_array(-128.0)),
    **valued_const_with_data('in_high', float32_array(127.0)),
    **valued_const_with_data('out_low', float32_array(-128.0)),
    **valued_const_with_data('out_high', float32_array(127.0)),
    **valued_const_with_data('non_const_in_low', float32_array(-128.0)),
    **valued_const_with_data('non_const_in_high', float32_array(127.0)),
    **valued_const_with_data('non_const_out_low', float32_array(-128.0)),
    **valued_const_with_data('non_const_out_high', float32_array(127.0)),
    **regular_op_with_shaped_data('input', [1, 2, 2], {'op': 'Parameter', 'type': 'Parameter'}),
    **regular_op_with_empty_data('const_quantize', {'op': 'QuantizeLinear'}),
    **regular_op_with_empty_data('non_const_quantize', {'op': 'QuantizeLinear'}),
    **regular_op_with_empty_data('const_dequantize', {'op': 'DequantizeLinear'}),
    **regular_op_with_empty_data('non_const_dequantize', {'op': 'DequantizeLinear'}),
    **regular_op_with_empty_data('add', {'op': 'Add'}),
    **regular_op_with_empty_data('mul_low', {'op': 'Mul'}),
    **regular_op_with_empty_data('mul_high', {'op': 'Mul'}),
    **regular_op_with_empty_data('const_fq', {'op': 'FakeQuantize'}),
    **regular_op_with_empty_data('const_cast', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.uint8}),
    **regular_op_with_empty_data('non_const_mul_low', {'op': 'Mul'}),
    **regular_op_with_empty_data('non_const_mul_high', {'op': 'Mul'}),
    **regular_op_with_empty_data('non_const_fq', {'op': 'FakeQuantize'}),
    **regular_op_with_empty_data('non_const_cast', {'op': 'Cast', 'type': 'Convert', 'dst_type': np.uint8}),
    **result('result'),

}


class QuantizeDequantizeLinearResolverTest(unittest.TestCase):
    def test_quantize_dequantize_linear_resolver(self):
        graph = build_graph(nodes_attrs=nodes_attributes,
                            edges=[
                                *connect('input', '0:non_const_quantize'),
                                *connect('y_scale_2', '1:non_const_quantize'),
                                *connect('y_zeropoint_2', '2:non_const_quantize'),
                                *connect('non_const_quantize', '0:non_const_dequantize'),
                                *connect('x_scale_2', '1:non_const_dequantize'),
                                *connect('x_zeropoint_2', '2:non_const_dequantize'),

                                *connect('const_input', '0:const_quantize'),
                                *connect('y_scale_1', '1:const_quantize'),
                                *connect('y_zeropoint_1', '2:const_quantize'),
                                *connect('const_quantize', '0:const_dequantize'),
                                *connect('x_scale_1', '1:const_dequantize'),
                                *connect('x_zeropoint_1', '2:const_dequantize'),
                                *connect('const_dequantize', '0:add'),
                                *connect('non_const_dequantize', '1:add'),
                                *connect('add', 'result')
                            ], nodes_with_edges_only=True)

        const_ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                      edges=[
                                          *connect('input', '0:non_const_quantize'),
                                          *connect('y_scale_2', '1:non_const_quantize'),
                                          *connect('y_zeropoint_2', '2:non_const_quantize'),
                                          *connect('non_const_quantize', '0:non_const_dequantize'),
                                          *connect('x_scale_2', '1:non_const_dequantize'),
                                          *connect('x_zeropoint_2', '2:non_const_dequantize'),

                                          *connect('const_input', '0:const_fq'),
                                          *connect('y_scale_1:0', '0:mul_low'),
                                          *connect('in_low', '1:mul_low'),
                                          ('y_scale_1_d', 'mul_high', {'out': 1, 'in': 0}),
                                          *connect('in_high', '1:mul_high'),
                                          *connect('mul_low', '1:const_fq'),
                                          *connect('mul_high', '2:const_fq'),
                                          *connect('out_low', '3:const_fq'),
                                          *connect('out_high', '4:const_fq'),
                                          *connect('const_fq', 'const_cast'),
                                          *connect('const_cast', '0:const_dequantize'),
                                          *connect('x_scale_1', '1:const_dequantize'),
                                          *connect('x_zeropoint_1', '2:const_dequantize'),
                                          *connect('const_dequantize', '0:add'),
                                          *connect('non_const_dequantize', '1:add'),
                                          *connect('add', 'result')
                                      ],nodes_with_edges_only=True)
        QuantizeDequantizeLinearResolver().find_and_replace_pattern(graph)
        graph.graph['layout'] = 'NCHW'
        (flag, resp) = compare_graphs(graph, const_ref_graph, 'result')
        self.assertTrue(flag, resp)

        ref_graph = build_graph(nodes_attrs=nodes_attributes,
                                edges=[
                                    *connect('input', '0:non_const_fq'),
                                    *connect('y_scale_2:0', '0:non_const_mul_low'),
                                    *connect('non_const_in_low', '1:non_const_mul_low'),
                                    ('y_scale_2_d', 'non_const_mul_high', {'out': 1, 'in': 0}),
                                    *connect('non_const_in_high', '1:non_const_mul_high'),
                                    *connect('non_const_mul_low', '1:non_const_fq'),
                                    *connect('non_const_mul_high', '2:non_const_fq'),
                                    *connect('non_const_out_low', '3:non_const_fq'),
                                    *connect('non_const_out_high', '4:non_const_fq'),
                                    *connect('non_const_fq', 'non_const_cast'),
                                    *connect('non_const_cast', '0:non_const_dequantize'),
                                    *connect('x_scale_2', '1:non_const_dequantize'),
                                    *connect('x_zeropoint_2', '2:non_const_dequantize'),

                                    *connect('const_input', '0:const_fq'),
                                    *connect('y_scale_1:0', '0:mul_low'),
                                    *connect('in_low', '1:mul_low'),
                                    ('y_scale_1_d', 'mul_high', {'out': 1, 'in': 0}),
                                    *connect('in_high', '1:mul_high'),
                                    *connect('mul_low', '1:const_fq'),
                                    *connect('mul_high', '2:const_fq'),
                                    *connect('out_low', '3:const_fq'),
                                    *connect('out_high', '4:const_fq'),
                                    *connect('const_fq', 'const_cast'),
                                    *connect('const_cast', '0:const_dequantize'),
                                    *connect('x_scale_1', '1:const_dequantize'),
                                    *connect('x_zeropoint_1', '2:const_dequantize'),
                                    *connect('const_dequantize', '0:add'),
                                    *connect('non_const_dequantize', '1:add'),
                                    *connect('add', 'result')
                                ], nodes_with_edges_only=True)
        QuantizeLinearResolver().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, ref_graph, 'result')
        self.assertTrue(flag, resp)
