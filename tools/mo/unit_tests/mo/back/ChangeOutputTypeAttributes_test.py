# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from copy import deepcopy

import numpy as np

from openvino.tools.mo.back.ChangeOutputTypeAttributes import ChangeOutputTypeAttributes
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.front.common.partial_infer.utils import float32_array
from openvino.tools.mo.middle.passes.convert_data_type import convert_blobs, data_type_str_to_np
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_empty_data, connect
from unit_tests.utils.graph import valued_const_with_data


class ChangeOutputTypeAttributesTests(unittest.TestCase):

    def test_range_correct_case(self):
        graph, graph_ref = build_range_test_graphs(start=0, limit=10, delta=1, dst_type_str='FP16')
        ChangeOutputTypeAttributes().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_range_correct_case_returns_shape_value(self):
        graph, graph_ref = build_range_test_graphs(start=0, limit=10, delta=1, dst_type_str='FP32',
                                                   src_type_str='FP16', returns_shape_value=True)
        ChangeOutputTypeAttributes().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)
        self.assertTrue(flag, resp)

    # starting from ~1000 FP16 absolute difference between neighbor values is more than 1
    # fails because of shape inconsistency
    def test_range_different_values(self):
        graph, graph_ref = build_range_test_graphs(start=0, limit=50000, delta=1, dst_type_str='FP16')
        self.assertRaises(Error, ChangeOutputTypeAttributes().find_and_replace_pattern, graph)

    def test_range_out_of_fp16_max(self):
        graph, graph_ref = build_range_test_graphs(start=0, limit=100000, delta=1, dst_type_str='FP16')
        self.assertRaises(Error, ChangeOutputTypeAttributes().find_and_replace_pattern, graph)

    def test_range_out_of_fp16_min(self):
        graph, graph_ref = build_range_test_graphs(start=0, limit=-100000, delta=-1, dst_type_str='FP16')
        self.assertRaises(Error, ChangeOutputTypeAttributes().find_and_replace_pattern, graph)

    def test_cast_correct_case(self):
        input_data = np.array([0, 1000, 4, 9, 0])
        graph, graph_ref = build_cast_test_graphs(input_data, dst_type_str='FP16')
        ChangeOutputTypeAttributes().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_cast_out_of_fp16_max(self):
        input_data = np.array([0, 100000, 4, 9, 0])
        graph, graph_ref = build_cast_test_graphs(input_data, dst_type_str='FP16')
        self.assertRaises(Error, ChangeOutputTypeAttributes().find_and_replace_pattern, graph)

    def test_cast_out_of_fp16_min(self):
        input_data = np.array([0, -100000, 4, 9, 0])
        graph, graph_ref = build_cast_test_graphs(input_data, dst_type_str='FP16')
        self.assertRaises(Error, ChangeOutputTypeAttributes().find_and_replace_pattern, graph)

    def test_cast_with_scalar(self):
        input_data = np.array(4)
        graph, graph_ref = build_cast_test_graphs(input_data, dst_type_str='FP16')
        ChangeOutputTypeAttributes().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)
        self.assertTrue(flag, resp)

def build_range_test_graphs(start=0, limit=10, delta=1, dst_type_str='FP16',
                            src_type_str='FP32', returns_shape_value=None):
    nodes = {
        **valued_const_with_data('start', float32_array(start)),
        **valued_const_with_data('limit', float32_array(limit)),
        **valued_const_with_data('delta', float32_array(delta)),
        **regular_op_with_empty_data('range', {'type': 'Range', 'op': 'Range',
                                               'returns_shape_value': returns_shape_value,
                                               'output_type': data_type_str_to_np(src_type_str),
                                               'infer': Range.infer}),
        **result('res'),
    }

    nodes_ref = deepcopy(nodes)
    nodes_ref.update({
        **regular_op_with_empty_data('range', {'type': 'Range', 'op': 'Range',
                                               'returns_shape_value': returns_shape_value,
                                               'output_type': data_type_str_to_np(dst_type_str),
                                               'infer': Range.infer}),
    })

    edges = [
        *connect('start', '0:range'),
        *connect('limit', '1:range'),
        *connect('delta', '2:range'),
        *connect('range', 'res'),
    ]
    graph = build_graph(nodes, edges)
    graph_ref = build_graph(nodes_ref, edges)

    graph = partial_infer(graph)

    graph.graph['cmd_params'].data_type = dst_type_str
    convert_blobs(graph, dst_type_str)
    return graph, graph_ref


def build_cast_test_graphs(input_data, dst_type_str='FP16'):
    nodes = {
        **valued_const_with_data('input', float32_array(input_data)),
        **regular_op_with_empty_data('cast', {'type': 'Convert', 'op': 'Cast',
                                              'dst_type': np.float32,
                                              'infer': Cast.infer}),
        **result('res'),
    }

    nodes_ref = deepcopy(nodes)
    nodes_ref.update({
        **regular_op_with_empty_data('cast', {'type': 'Convert', 'op': 'Cast',
                                              'dst_type': data_type_str_to_np(dst_type_str),
                                              'infer': Cast.infer}),
    })

    edges = [
        *connect('input', 'cast'),
        *connect('cast', 'res'),
    ]
    graph = build_graph(nodes, edges)
    graph_ref = build_graph(nodes_ref, edges)

    graph = partial_infer(graph)

    graph.graph['cmd_params'].data_type = dst_type_str
    convert_blobs(graph, dst_type_str)
    return graph, graph_ref
