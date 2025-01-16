# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.kaldi.replace_timeheightconvolution import ReplaceTimeHeightConvolutionPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op, connect_front, const


class TimeheightconvolutionReplacerTest(unittest.TestCase):
    nodes = {
        **regular_op('placeholder', {}),
        **regular_op('timeheightconv', {'op': 'timeheightconvolutioncomponent'}),
        **const('weights', int64_array([])),
        **const('biases', int64_array([])),
        **regular_op('placeholder_out', {}),

        **regular_op('concat', {'type': 'Concat', 'axis': 1}),
        **regular_op('memoryoffset_0', {'type': None, 'op': 'MemoryOffset', 't': -1, 'has_default': False}),
        **regular_op('memoryoffset_1', {'type': None, 'op': 'MemoryOffset', 't': 0, 'has_default': False}),
        **regular_op('memoryoffset_2', {'type': None, 'op': 'MemoryOffset', 't': 1, 'has_default': True}),
        **regular_op('conv', {'op': 'Convolution', 'type': 'Convolution', 'output': 12, 'height_in': 80}),
    }

    def test_timeheightconvolution_1offset(self):
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:timeheightconv'),
            *connect_front('weights', '1:timeheightconv'),
            *connect_front('biases', '2:timeheightconv'),
            *connect_front('timeheightconv', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        conv = graph.nodes['timeheightconv']
        conv['height_subsample'] = 1
        conv['height_in'] = 80
        conv['height_out'] = 80
        conv['in_channels'] = 1
        conv['out_channels'] = 12
        conv['offsets'] = int64_array([[-1, -1], [-1, 0], [-1, 1]])
        conv['time_offsets'] = [-1]
        graph.nodes['weights']['value'] = np.zeros([36])

        ref_graph = build_graph(self.nodes, [
            *connect_front('placeholder', 'memoryoffset_0'),
            *connect_front('memoryoffset_0', '0:concat'),
            *connect_front('concat', '0:conv'),
            *connect_front('weights', '1:conv'),
            *connect_front('biases', '2:conv'),
            *connect_front('conv', 'placeholder_out')
        ], nodes_with_edges_only=True)
        ref_graph.nodes['weights']['value'] = np.zeros([36])
        new_conv = ref_graph.nodes['conv']
        new_conv['pad'] = int64_array([[0, 0], [0, 0], [0, 0], [1, 1]])
        new_conv['dilation'] = int64_array([1, 1, 1, 1])
        new_conv['kernel'] = int64_array([12, 1, 1, 3])
        new_conv['stride'] = int64_array([1, 1, 1, 1])


        ReplaceTimeHeightConvolutionPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_timeheightconvolution_2_offsets(self):
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:timeheightconv'),
            *connect_front('weights', '1:timeheightconv'),
            *connect_front('biases', '2:timeheightconv'),
            *connect_front('timeheightconv', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        conv = graph.nodes['timeheightconv']
        conv['height_subsample'] = 1
        conv['height_in'] = 80
        conv['height_out'] = 80
        conv['in_channels'] = 1
        conv['out_channels'] = 12
        conv['offsets'] = int64_array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1]])
        conv['time_offsets'] = int64_array([-1, 0])
        graph.nodes['weights']['value'] = np.zeros([72])

        ref_graph = build_graph(self.nodes, [
            *connect_front('placeholder', 'memoryoffset_0'),
            *connect_front('placeholder', 'memoryoffset_1'),
            *connect_front('memoryoffset_0', '0:concat'),
            *connect_front('memoryoffset_1', '1:concat'),
            *connect_front('concat', '0:conv'),
            *connect_front('weights', '1:conv'),
            *connect_front('biases', '2:conv'),
            *connect_front('conv', 'placeholder_out')
        ], nodes_with_edges_only=True)
        ref_graph.nodes['weights']['value'] = np.zeros([72])
        new_conv = ref_graph.nodes['conv']
        new_conv['pad'] = int64_array([[0, 0], [0, 0], [0, 0], [1, 1]])
        new_conv['dilation'] = int64_array([1, 1, 1, 1])
        new_conv['kernel'] = int64_array([12, 1, 2, 3])
        new_conv['stride'] = int64_array([1, 1, 1, 1])

        ReplaceTimeHeightConvolutionPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_timeheightconvolution_2_offsets_def(self):
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:timeheightconv'),
            *connect_front('weights', '1:timeheightconv'),
            *connect_front('biases', '2:timeheightconv'),
            *connect_front('timeheightconv', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        conv = graph.nodes['timeheightconv']
        conv['height_subsample'] = 1
        conv['height_in'] = 80
        conv['height_out'] = 80
        conv['in_channels'] = 1
        conv['out_channels'] = 12
        conv['offsets'] = int64_array([[0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])
        conv['time_offsets'] = int64_array([0])
        graph.nodes['weights']['value'] = np.zeros([72])

        ref_graph = build_graph(self.nodes, [
            *connect_front('placeholder', 'memoryoffset_1'),
            *connect_front('placeholder', 'memoryoffset_2'),
            *connect_front('memoryoffset_1', '0:concat'),
            *connect_front('memoryoffset_2', '1:concat'),
            *connect_front('concat', '0:conv'),
            *connect_front('weights', '1:conv'),
            *connect_front('biases', '2:conv'),
            *connect_front('conv', 'placeholder_out')
        ], nodes_with_edges_only=True)
        ref_graph.nodes['weights']['value'] = np.zeros([72])
        new_conv = ref_graph.nodes['conv']
        new_conv['pad'] = int64_array([[0, 0], [0, 0], [0, 0], [1, 1]])
        new_conv['dilation'] = int64_array([1, 1, 1, 1])
        new_conv['kernel'] = int64_array([12, 1, 2, 3])
        new_conv['stride'] = int64_array([1, 1, 1, 1])

        ReplaceTimeHeightConvolutionPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_timeheightconvolution_2_offsets_dilation(self):
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:timeheightconv'),
            *connect_front('weights', '1:timeheightconv'),
            *connect_front('biases', '2:timeheightconv'),
            *connect_front('timeheightconv', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        conv = graph.nodes['timeheightconv']
        conv['height_subsample'] = 1
        conv['height_in'] = 80
        conv['height_out'] = 80
        conv['in_channels'] = 1
        conv['out_channels'] = 12
        conv['offsets'] = int64_array([[-1, -3], [-1, 0], [-1, 3], [1, -3], [1, 0], [1, 3]])
        conv['time_offsets'] = int64_array([-1])
        graph.nodes['weights']['value'] = np.zeros([72])

        ref_graph = build_graph(self.nodes, [
            *connect_front('placeholder', 'memoryoffset_0'),
            *connect_front('placeholder', 'memoryoffset_2'),
            *connect_front('memoryoffset_0', '0:concat'),
            *connect_front('memoryoffset_2', '1:concat'),
            *connect_front('concat', '0:conv'),
            *connect_front('weights', '1:conv'),
            *connect_front('biases', '2:conv'),
            *connect_front('conv', 'placeholder_out')
        ], nodes_with_edges_only=True)
        ref_graph.nodes['weights']['value'] = np.zeros([72])
        new_conv = ref_graph.nodes['conv']
        new_conv['pad'] = int64_array([[0, 0], [0, 0], [0, 0], [3, 3]])
        new_conv['dilation'] = int64_array([1, 1, 2, 3])
        new_conv['kernel'] = int64_array([12, 1, 2, 3])
        new_conv['stride'] = int64_array([1, 1, 1, 1])

        ReplaceTimeHeightConvolutionPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_timeheightconvolution_2_offsets_pad(self):
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:timeheightconv'),
            *connect_front('weights', '1:timeheightconv'),
            *connect_front('biases', '2:timeheightconv'),
            *connect_front('timeheightconv', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'
        conv = graph.nodes['timeheightconv']
        conv['height_subsample'] = 1
        conv['height_in'] = 80
        conv['height_out'] = 74
        conv['in_channels'] = 1
        conv['out_channels'] = 12
        conv['offsets'] = int64_array([[-1, 0], [-1, 3], [-1, 6], [1, 0], [1, 3], [1, 6]])
        conv['time_offsets'] = int64_array([-1])
        graph.nodes['weights']['value'] = np.zeros([72])

        ref_graph = build_graph(self.nodes, [
            *connect_front('placeholder', 'memoryoffset_0'),
            *connect_front('placeholder', 'memoryoffset_2'),
            *connect_front('memoryoffset_0', '0:concat'),
            *connect_front('memoryoffset_2', '1:concat'),
            *connect_front('concat', '0:conv'),
            *connect_front('weights', '1:conv'),
            *connect_front('biases', '2:conv'),
            *connect_front('conv', 'placeholder_out')
        ], nodes_with_edges_only=True)
        ref_graph.nodes['weights']['value'] = np.zeros([72])
        new_conv = ref_graph.nodes['conv']
        new_conv['pad'] = int64_array([[0, 0], [0, 0], [0, 0], [0, 0]])
        new_conv['dilation'] = int64_array([1, 1, 2, 3])
        new_conv['kernel'] = int64_array([12, 1, 2, 3])
        new_conv['stride'] = int64_array([1, 1, 1, 1])

        ReplaceTimeHeightConvolutionPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_timeheightconvolution_out_channels(self):
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:timeheightconv'),
            *connect_front('weights', '1:timeheightconv'),
            *connect_front('biases', '2:timeheightconv'),
            *connect_front('timeheightconv', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'
        conv = graph.nodes['timeheightconv']
        conv['height_subsample'] = 1
        conv['height_in'] = 80
        conv['height_out'] = 74
        conv['in_channels'] = 3
        conv['out_channels'] = 4
        conv['offsets'] = int64_array([[-1, 0], [-1, 3], [-1, 6], [1, 0], [1, 3], [1, 6]])
        conv['time_offsets'] = int64_array([-1])
        graph.nodes['weights']['value'] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                                    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
                                                    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72])

        ref_graph = build_graph(self.nodes, [
            *connect_front('placeholder', 'memoryoffset_0'),
            *connect_front('placeholder', 'memoryoffset_2'),
            *connect_front('memoryoffset_0', '0:concat'),
            *connect_front('memoryoffset_2', '1:concat'),
            *connect_front('concat', '0:conv'),
            *connect_front('weights', '1:conv'),
            *connect_front('biases', '2:conv'),
            *connect_front('conv', 'placeholder_out')
        ], nodes_with_edges_only=True)
        ref_graph.nodes['weights']['value'] = np.array([1, 4, 7, 10, 13, 16, 2, 5, 8, 11, 14, 17, 3, 6, 9, 12, 15, 18,
                                                        19, 22, 25, 28, 31, 34, 20, 23, 26, 29, 32, 35, 21, 24, 27, 30, 33, 36,
                                                        37, 40, 43, 46, 49, 52, 38, 41, 44, 47, 50, 53, 39, 42, 45, 48, 51, 54,
                                                        55, 58, 61, 64, 67, 70, 56, 59, 62, 65, 68, 71, 57, 60, 63, 66, 69, 72])
        new_conv = ref_graph.nodes['conv']
        new_conv['output'] = 4
        new_conv['pad'] = int64_array([[0, 0], [0, 0], [0, 0], [0, 0]])
        new_conv['dilation'] = int64_array([1, 1, 2, 3])
        new_conv['kernel'] = int64_array([4, 3, 2, 3])
        new_conv['stride'] = int64_array([1, 1, 1, 1])

        ReplaceTimeHeightConvolutionPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_timeheightconvolution_2_offsets_stride(self):
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:timeheightconv'),
            *connect_front('weights', '1:timeheightconv'),
            *connect_front('biases', '2:timeheightconv'),
            *connect_front('timeheightconv', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'
        conv = graph.nodes['timeheightconv']
        conv['height_subsample'] = 2
        conv['height_in'] = 80
        conv['height_out'] = 37
        conv['in_channels'] = 1
        conv['out_channels'] = 12
        conv['offsets'] = int64_array([[-1, 0], [-1, 3], [-1, 6], [1, 0], [1, 3], [1, 6]])
        conv['time_offsets'] = int64_array([-1])
        graph.nodes['weights']['value'] = np.zeros([72])

        ref_graph = build_graph(self.nodes, [
            *connect_front('placeholder', 'memoryoffset_0'),
            *connect_front('placeholder', 'memoryoffset_2'),
            *connect_front('memoryoffset_0', '0:concat'),
            *connect_front('memoryoffset_2', '1:concat'),
            *connect_front('concat', '0:conv'),
            *connect_front('weights', '1:conv'),
            *connect_front('biases', '2:conv'),
            *connect_front('conv', 'placeholder_out')
        ], nodes_with_edges_only=True)
        ref_graph.nodes['weights']['value'] = np.zeros([72])
        new_conv = ref_graph.nodes['conv']
        new_conv['pad'] = int64_array([[0, 0], [0, 0], [0, 0], [0, 0]])
        new_conv['dilation'] = int64_array([1, 1, 2, 3])
        new_conv['kernel'] = int64_array([12, 1, 2, 3])
        new_conv['stride'] = int64_array([1, 1, 1, 2])

        ReplaceTimeHeightConvolutionPattern().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)
