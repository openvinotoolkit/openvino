# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.load.tf.loader import graph_or_sub_graph_has_nhwc_ops
from unit_tests.utils.graph import build_graph, result, regular_op, const, connect_front


class TFLoaderTest(unittest.TestCase):
    @staticmethod
    def build_conv_graph():
        nodes = {
            **const('weights', np.random.randn(1, 1, 1, 1)),
            **regular_op('input', {'op': 'Parameter'}),
            **regular_op('conv', {'op': 'Conv2D', 'layout': 'NHWC'}),
            **result('result'),
        }
        edges = [*connect_front('input', '0:conv'),
                 *connect_front('weights', '1:conv'),
                 *connect_front('conv:0', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        graph.stage = 'front'
        return graph

    @staticmethod
    def build_parameter_result_graph():
        nodes = {
            **regular_op('input', {'op': 'Parameter'}),
            **result('result'),
        }
        edges = [*connect_front('input', '0:result'),
                 ]
        graph = build_graph(nodes, edges)
        graph.stage = 'front'
        return graph

    @staticmethod
    def build_loop_graph(body_graph):
        # create fake Loop operation
        nodes = {
            **regular_op('input', {'op': 'Parameter'}),
            **regular_op('loop', {'op': 'Loop', 'body': body_graph, 'sub_graphs': ['body']}),
            **result('result'),
        }
        edges = [*connect_front('input', '0:loop'),
                 *connect_front('loop:0', 'result'),
                 ]
        graph = build_graph(nodes, edges)
        graph.stage = 'front'
        return graph

    def test_convolution_main_graph(self):
        self.assertTrue(graph_or_sub_graph_has_nhwc_ops(self.build_conv_graph()))

    def test_convolution_loop_body_graph(self):
        self.assertTrue(graph_or_sub_graph_has_nhwc_ops(self.build_loop_graph(self.build_conv_graph())))

    def test_no_convolution_main_graph(self):
        self.assertFalse(graph_or_sub_graph_has_nhwc_ops(self.build_parameter_result_graph()))

    def test_no_convolution_main_and_sub_graph(self):
        self.assertFalse(graph_or_sub_graph_has_nhwc_ops(self.build_loop_graph(self.build_parameter_result_graph())))
