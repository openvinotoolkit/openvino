# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from extensions.front.deduce_parameter_pshape import DeduceParameterShape
from mo.front.common.partial_infer.utils import dynamic_dimension, strict_compare_tensors, undefined_shape_of_rank
from mo.graph.graph import Node
from unit_tests.utils.graph import build_graph


class TestDeduceParameterPShapeFromConvolution(unittest.TestCase):

    @staticmethod
    def build_graph_and_deduce_pshape(num_channels=3, conv_dim_size=2, direct_connection_to_conv=False):
        kernel_size = 4
        kernel_shape = [kernel_size] * conv_dim_size
        weights_value = np.zeros((16, num_channels, *kernel_shape), dtype='float32')

        nodes = {
            'input': {'kind': 'op', 'op': 'Parameter', 'type': 'Parameter'},

            'add_1': {'kind': 'op', 'op': 'Add'},
            'add_1_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'mul_1': {'kind': 'op', 'op': 'Mul'},
            'mul_1_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'weights_1': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': weights_value},
            'conv_1': {'kind': 'op', 'op': 'Convolution', 'type': 'Convolution'},

            'add_2': {'kind': 'op', 'op': 'Add'},
            'add_2_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'mul_2': {'kind': 'op', 'op': 'Mul'},
            'mul_2_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'weights_2': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'conv_2': {'kind': 'op', 'op': 'Convolution', 'type': 'Convolution'},

            'res': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
            'res_2': {'kind': 'op', 'op': 'Result', 'type': 'Result'}
        }

        edges = [
            ('input', 'add_1', {'in': 0}) if not direct_connection_to_conv else ('input', 'conv_1', {'in': 0}),
            ('add_1_const', 'add_1', {'in': 1}),

            ('add_1', 'mul_1', {'in': 0}),
            ('mul_1_const', 'mul_1', {'in': 1}),

            ('mul_1', 'conv_1', {'in': 0}) if not direct_connection_to_conv else ('mul_1', 'res_2', {'in': 0}),
            ('weights_1', 'conv_1', {'in': 1}),

            ('conv_1', 'add_2', {'in': 0}),
            ('add_2_const', 'add_2', {'in': 1}),
            ('add_2', 'mul_2', {'in': 0}),
            ('mul_2_const', 'mul_2', {'in': 1}),

            ('mul_2', 'conv_2', {'in': 0}),
            ('weights_2', 'conv_2', {'in': 1}),

            ('conv_2', 'res')
        ]

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'front'
        graph.graph['fw'] = 'onnx'
        graph.graph['layout'] = 'NCHW'

        DeduceParameterShape().find_and_replace_pattern(graph)
        parameter = Node(graph, 'input')
        return parameter['shape']

    def test_parameter_conv_ordinally_connected_1(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_channels=3, conv_dim_size=2)
        assert strict_compare_tensors(parameter_shape, [dynamic_dimension, 3, dynamic_dimension, dynamic_dimension])

    def test_parameter_conv_ordinally_connected_2(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_channels=1, conv_dim_size=2)
        assert strict_compare_tensors(parameter_shape, [dynamic_dimension, 1, dynamic_dimension, dynamic_dimension])

    def test_parameter_conv_ordinally_connected_3(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_channels=3, conv_dim_size=1)
        assert strict_compare_tensors(parameter_shape, [dynamic_dimension, 3, dynamic_dimension])

    def test_parameter_conv_ordinally_connected_4(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_channels=3, conv_dim_size=3)
        assert strict_compare_tensors(parameter_shape, [dynamic_dimension, 3, dynamic_dimension, dynamic_dimension, dynamic_dimension])

    def test_parameter_conv_directly_connected_1(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_channels=3, conv_dim_size=2,
                                                             direct_connection_to_conv=True)
        assert strict_compare_tensors(parameter_shape, [dynamic_dimension, 3, dynamic_dimension, dynamic_dimension])

    def test_parameter_conv_directly_connected_2(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_channels=1, conv_dim_size=2,
                                                             direct_connection_to_conv=True)
        assert strict_compare_tensors(parameter_shape, [dynamic_dimension, 1, dynamic_dimension, dynamic_dimension])


class TestDeduceParameterPShapeFromPooling(unittest.TestCase):

    @staticmethod
    def build_graph_and_deduce_pshape(num_spatial_dims=2, direct_connection=False):
        kernel_size = 4
        kernel_shape = [kernel_size] * num_spatial_dims
        window = [1, 1, *kernel_shape]

        nodes = {
            'input': {'kind': 'op', 'op': 'Parameter', 'type': 'Parameter'},

            'add_1': {'kind': 'op', 'op': 'Add'},
            'add_1_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'mul_1': {'kind': 'op', 'op': 'Mul'},
            'mul_1_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'weights_1': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': window},
            'pool_1': {'kind': 'op', 'op': 'Pooling', 'type': 'Pooling'},

            'add_2': {'kind': 'op', 'op': 'Add'},
            'add_2_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'mul_2': {'kind': 'op', 'op': 'Mul'},
            'mul_2_const': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'weights_2': {'kind': 'op', 'op': 'Const', 'type': 'Const', 'value': None},
            'pool_2': {'kind': 'op', 'op': 'Pooling', 'type': 'Pooling'},

            'res': {'kind': 'op', 'op': 'Result', 'type': 'Result'},
            'res_2': {'kind': 'op', 'op': 'Result', 'type': 'Result'}
        }

        edges = [
            ('input', 'add_1', {'in': 0}) if not direct_connection else ('input', 'pool_1', {'in': 0}),
            ('add_1_const', 'add_1', {'in': 1}),

            ('add_1', 'mul_1', {'in': 0}),
            ('mul_1_const', 'mul_1', {'in': 1}),

            ('mul_1', 'pool_1', {'in': 0}) if not direct_connection else ('mul_1', 'res_2', {'in': 0}),
            ('weights_1', 'pool_1', {'in': 1}),

            ('pool_1', 'add_2', {'in': 0}),
            ('add_2_const', 'add_2', {'in': 1}),
            ('add_2', 'mul_2', {'in': 0}),
            ('mul_2_const', 'mul_2', {'in': 1}),

            ('mul_2', 'pool_2', {'in': 0}),
            ('weights_2', 'pool_2', {'in': 1}),

            ('pool_2', 'res')
        ]

        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph.stage = 'front'
        graph.graph['fw'] = 'onnx'
        graph.graph['layout'] = 'NCHW'

        DeduceParameterShape().find_and_replace_pattern(graph)
        parameter = Node(graph, 'input')
        return parameter['shape']

    def test_parameter_conv_ordinally_connected_1(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_spatial_dims=2)
        assert strict_compare_tensors(parameter_shape, undefined_shape_of_rank(4))

    def test_parameter_conv_ordinally_connected_2(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_spatial_dims=2)
        assert strict_compare_tensors(parameter_shape, undefined_shape_of_rank(4))

    def test_parameter_conv_ordinally_connected_3(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_spatial_dims=1)
        assert strict_compare_tensors(parameter_shape, undefined_shape_of_rank(3))

    def test_parameter_conv_ordinally_connected_4(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_spatial_dims=3)
        assert strict_compare_tensors(parameter_shape, undefined_shape_of_rank(5))

    def test_parameter_conv_directly_connected_1(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_spatial_dims=2,
                                                             direct_connection=True)
        assert strict_compare_tensors(parameter_shape, undefined_shape_of_rank(4))

    def test_parameter_conv_directly_connected_2(self):
        parameter_shape = self.build_graph_and_deduce_pshape(num_spatial_dims=2,
                                                             direct_connection=True)
        assert strict_compare_tensors(parameter_shape, undefined_shape_of_rank(4))
