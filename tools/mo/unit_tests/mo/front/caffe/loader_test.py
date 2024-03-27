# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from google.protobuf import text_format

from openvino.tools.mo.front.caffe.loader import caffe_pb_to_nx
from openvino.tools.mo.front.caffe.proto import caffe_pb2
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry

proto_str_one_input = 'name: "network" ' \
                      'layer { ' \
                      'name: "Input0" ' \
                      'type: "Input" ' \
                      'top: "Input0" ' \
                      'input_param { ' \
                      'shape: { ' \
                      'dim: 1 ' \
                      'dim: 3 ' \
                      'dim: 224 ' \
                      'dim: 224 ' \
                      '} ' \
                      '} ' \
                      '}'

proto_str_old_styled_multi_input = 'name: "network" ' \
                                   'input: "Input0" ' \
                                   'input_dim: 1 ' \
                                   'input_dim: 3 ' \
                                   'input_dim: 224 ' \
                                   'input_dim: 224 ' \
                                   'input: "data" ' \
                                   'input_dim: 1 ' \
                                   'input_dim: 3 '

proto_str_input = 'name: "network" ' \
                  'input: "data" ' \
                  'input_shape ' \
                  '{ ' \
                  'dim: 1 ' \
                  'dim: 3 ' \
                  'dim: 224 ' \
                  'dim: 224 ' \
                  '}'

proto_str_multi_input = 'name: "network" ' \
                        'input: "data" ' \
                        'input_shape ' \
                        '{ ' \
                        'dim: 1 ' \
                        'dim: 3 ' \
                        'dim: 224 ' \
                        'dim: 224 ' \
                        '} ' \
                        'input: "data1" ' \
                        'input_shape ' \
                        '{ ' \
                        'dim: 1 ' \
                        'dim: 3 ' \
                        '}'

proto_str_old_styled_input = 'name: "network" ' \
                             'input: "data" ' \
                             'input_dim: 1 ' \
                             'input_dim: 3 ' \
                             'input_dim: 224 ' \
                             'input_dim: 224 '

layer_proto_str = 'layer { ' \
                  'name: "conv1" ' \
                  'type: "Convolution" ' \
                  'bottom: "data" ' \
                  'top: "conv1" ' \
                  '}'

proto_same_name_layers = 'layer { ' \
                         'name: "conv1" ' \
                         'type: "Convolution" ' \
                         'bottom: "data" ' \
                         'top: "conv1" ' \
                         '} ' \
                         'layer { ' \
                         'name: "conv1" ' \
                         'type: "Convolution" ' \
                         'bottom: "data1" ' \
                         'top: "conv1_2" ' \
                         '}'

class TestLoader(UnitTestWithMockedTelemetry):
    def test_caffe_pb_to_nx_one_input(self):
        proto = caffe_pb2.NetParameter()
        text_format.Merge(proto_str_one_input, proto)
        input_shapes = caffe_pb_to_nx(Graph(), proto, None)
        expected_input_shapes = {
            'Input0': np.array([1, 3, 224, 224])
        }

        for i in expected_input_shapes:
            np.testing.assert_array_equal(input_shapes[i], expected_input_shapes[i])

    def test_caffe_pb_to_nx_old_styled_multi_input(self):
        proto = caffe_pb2.NetParameter()
        text_format.Merge(proto_str_old_styled_multi_input + layer_proto_str, proto)
        self.assertRaises(Error, caffe_pb_to_nx, Graph(), proto, None)

    def test_caffe_pb_to_nx_old_styled_input(self):
        proto = caffe_pb2.NetParameter()
        text_format.Merge(proto_str_old_styled_input + layer_proto_str, proto)
        input_shapes = caffe_pb_to_nx(Graph(), proto, None)
        expected_input_shapes = {
            'data': np.array([1, 3, 224, 224])
        }

        for i in expected_input_shapes:
            np.testing.assert_array_equal(input_shapes[i], expected_input_shapes[i])

    def test_caffe_pb_to_standart_input(self):
        proto = caffe_pb2.NetParameter()
        text_format.Merge(proto_str_input + layer_proto_str, proto)
        input_shapes = caffe_pb_to_nx(Graph(), proto, None)
        expected_input_shapes = {
            'data': np.array([1, 3, 224, 224])
        }

        for i in expected_input_shapes:
            np.testing.assert_array_equal(input_shapes[i], expected_input_shapes[i])

    def test_caffe_pb_to_multi_input(self):
        proto = caffe_pb2.NetParameter()
        text_format.Merge(proto_str_multi_input + layer_proto_str, proto)
        input_shapes = caffe_pb_to_nx(Graph(), proto, None)
        expected_input_shapes = {
            'data': np.array([1, 3, 224, 224]),
            'data1': np.array([1, 3])
        }

        for i in expected_input_shapes:
            np.testing.assert_array_equal(input_shapes[i], expected_input_shapes[i])

    def test_caffe_same_name_layer(self):
        proto = caffe_pb2.NetParameter()
        text_format.Merge(proto_str_multi_input + proto_same_name_layers, proto)
        graph = Graph()
        caffe_pb_to_nx(graph, proto, None)
        # 6 nodes because: 2 inputs + 2 convolutions + 2 identity nodes used as fake outputs
        np.testing.assert_equal(len(graph.nodes()), 6)
