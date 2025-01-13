# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np
import openvino.runtime as ov
import pytest
from openvino.runtime import PartialShape, Type, Dimension

from common.mo_convert_test_class import CommonMOConvertTest
from common.utils.tf_utils import save_to_pb


class TestComplexParams(CommonMOConvertTest):
    @staticmethod
    def create_tf_model(tmp_dir):
        #
        #   Create Tensorflow model with multiple inputs/outputs
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp1 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input1')
            inp2 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input2')
            inp3 = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input3')

            relu1 = tf.nn.relu(inp1, name='Relu1')
            relu2 = tf.nn.relu(inp2, name='Relu2')
            relu3 = tf.nn.relu(inp3, name='Relu3')

            concat = tf.concat([relu1, relu2, relu3], axis=0, name='Concat')

            outputs = tf.split(concat, 3)
            outputs_list = []
            for i, output in enumerate(outputs):
                outputs_list.append(tf.nn.sigmoid(output, name='Sigmoid_{}'.format(i)))

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir)

    @staticmethod
    def create_tf_model_single_input_output(tmp_dir):
        #
        #   Create Tensorflow model with single input/output
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            inp = tf.compat.v1.placeholder(tf.float32, [1, 3, 2, 2], 'Input')

            relu = tf.nn.relu(inp, name='Relu')

            output = tf.nn.sigmoid(relu, name='Sigmoid')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # save model to .pb and return path to the model
        return save_to_pb(tf_net, tmp_dir)

    test_data = [
        {'params_test': {'input': {"Input:0": [3, 2]}},
         'params_ref': {'input': "Input:0[3,2]"}},
        {'params_test': {'input': (3, 2)},
         'params_ref': {'input': "Input:0[3,2]"}},
        {'params_test': {'input': (3, Dimension(2))},
         'params_ref': {'input': "Input:0[3,2]"}},
        {'params_test': {'input': (-1, 10)},
         'params_ref': {'input': "Input:0[?,10]"}},
        {'params_test': {'input': PartialShape([-1, 10])},
         'params_ref': {'input': "Input:0[?,10]"}},
        {'params_test': {'input': [Dimension(3, 10), 10, -1]},
         'params_ref': {'input': 'Input:0[3..10,10,?]'}},
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_tf_model_single_input_output(self, params, ie_device, precision, ir_version,
                                                     temp_dir, use_legacy_frontend):
        tf_net_path = self.create_tf_model_single_input_output(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

    @staticmethod
    def create_onnx_model_with_comma_in_names(temp_dir):
        import onnx
        from onnx import helper
        from onnx import TensorProto

        input_1 = helper.make_tensor_value_info('input_1', TensorProto.FLOAT, [1, 3, 2, 2])
        input_2 = helper.make_tensor_value_info('input_2', TensorProto.FLOAT, [1, 3, 2, 2])
        output = helper.make_tensor_value_info('relu_1,relu_2', TensorProto.FLOAT, [1, 3, 4, 2])

        node_def_1 = onnx.helper.make_node(
            'Relu',
            inputs=['input_1'],
            outputs=['Relu_1_data'],
            name='relu_1'
        )
        node_def_2 = onnx.helper.make_node(
            'Relu',
            inputs=['input_2'],
            outputs=['Relu_2_data'],
            name='relu_2'
        )
        node_def_3 = onnx.helper.make_node(
            'Concat',
            inputs=['Relu_1_data', 'Relu_2_data'],
            outputs=['relu_1,relu_2'],
            axis=3,
        )

        graph_def = helper.make_graph(
            [node_def_1, node_def_2, node_def_3],
            'test_model',
            [input_1, input_2],
            [output],
        )
        onnx_net = helper.make_model(graph_def, producer_name='test_model')
        model_path = temp_dir + '/test_model.onnx'
        onnx.save(onnx_net, model_path)
        return model_path

    @staticmethod
    def create_ref_graph_with_comma_in_names():
        from openvino.runtime.opset12 import relu, concat
        from openvino.runtime.op import Parameter
        import openvino as ov

        parameter1 = Parameter(ov.Type.f32, ov.Shape([1, 3, 2, 2]))
        parameter2 = Parameter(ov.Type.f32, ov.Shape([1, 3, 2, 2]))
        relu_1 = relu(parameter1)
        relu_2 = relu(parameter2)

        output = concat([relu_1, relu_2], 3)
        return ov.Model([output], [parameter1, parameter2])

    @staticmethod
    def create_onnx_model_with_several_outputs(temp_dir):
        import onnx
        from onnx import helper
        from onnx import TensorProto

        shape = [1, 3, 2, 2]

        input_1 = helper.make_tensor_value_info('input_1', TensorProto.FLOAT, shape)
        input_2 = helper.make_tensor_value_info('input_2', TensorProto.FLOAT, shape)
        concat_output = helper.make_tensor_value_info('concat', TensorProto.FLOAT, shape)
        relu_output = helper.make_tensor_value_info('Relu_1_data', TensorProto.FLOAT, shape)

        node_def_1 = onnx.helper.make_node(
            'Relu',
            inputs=['input_1'],
            outputs=['Relu_1_data'],
            name='relu_1'
        )
        node_def_2 = onnx.helper.make_node(
            'Relu',
            inputs=['input_2'],
            outputs=['Relu_2_data'],
            name='relu_2'
        )
        node_def_3 = onnx.helper.make_node(
            'Concat',
            inputs=['Relu_1_data', 'Relu_2_data'],
            outputs=['concat'],
            axis=3,
        )

        graph_def = helper.make_graph(
            [node_def_1, node_def_2, node_def_3],
            'test_model',
            [input_1, input_2],
            [relu_output, concat_output],
        )
        onnx_net = helper.make_model(graph_def, producer_name='test_model')
        model_path = temp_dir + '/test_model.onnx'
        onnx.save(onnx_net, model_path)
        return model_path

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ovc_convert_model_with_comma_in_names(self, ie_device, precision, ir_version,
                                                  temp_dir, use_legacy_frontend):
        onnx_net_path = self.create_onnx_model_with_comma_in_names(temp_dir)
        ref_model = self.create_ref_graph_with_comma_in_names()
        test_params = {'input_model': onnx_net_path, 'output': 'relu_1,relu_2'}

        self._test_by_ref_graph(temp_dir, test_params, ref_model, compare_tensor_names=False)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ovc_convert_model_with_several_output(self, ie_device, precision, ir_version,
                                                  temp_dir, use_legacy_frontend):
        onnx_net_path = self.create_onnx_model_with_several_outputs(temp_dir)
        convert_model_params = {'input_model': onnx_net_path, 'output': ['Relu_1_data', 'concat']}
        cli_tool_params = {'input_model': onnx_net_path, 'output': 'Relu_1_data,concat'}

        self._test(temp_dir, convert_model_params, cli_tool_params)


    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_non_numpy_types(self, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        import tensorflow as tf
        def func(a, b):
            return [a, b]
        model = tf.function(func, input_signature=[tf.TensorSpec([2], tf.float32, "a"),
                                                   tf.TensorSpec([2], tf.float32, "b")])
        parameter1 = ov.opset8.parameter(ov.Shape([2]), ov.Type.bf16)
        parameter2 = ov.opset8.parameter(ov.Shape([2]), ov.Type.bf16)
        bf16_ref = ov.Model([parameter1, parameter2], [parameter1, parameter2])

        parameter1 = ov.opset8.parameter(ov.Shape([2]), ov.Type.string)
        parameter2 = ov.opset8.parameter(ov.Shape([2]), ov.Type.string)
        string_ref = ov.Model([parameter1, parameter2], [parameter1, parameter2])

        self._test_by_ref_graph(temp_dir, {'input_model': model, 'input': [ov.Type.bf16, tf.bfloat16]}, bf16_ref, compare_tensor_names=False)
        self._test_by_ref_graph(temp_dir, {'input_model': model, 'input': {'a': ov.Type.bf16, 'b': tf.bfloat16}}, bf16_ref, compare_tensor_names=False)
        self._test_by_ref_graph(temp_dir, {'input_model': model, 'input': [ov.Type.string, tf.string]}, string_ref, compare_tensor_names=False)
        self._test_by_ref_graph(temp_dir, {'input_model': model, 'input': {'a': ov.Type.string, 'b': tf.string}}, string_ref, compare_tensor_names=False)

class NegativeCases(unittest.TestCase):
    test_directory = os.path.dirname(os.path.realpath(__file__))

    def test_input_output_cut_exceptions(self):
        from openvino.tools.ovc import convert_model
        with tempfile.TemporaryDirectory(dir=self.test_directory) as temp_dir:
            tf_net_path = TestComplexParams.create_tf_model_single_input_output(temp_dir)

            with self.assertRaisesRegex(Exception, ".*Name Relu is not found among model inputs.*"):
                convert_model(tf_net_path, input='Relu')
            with self.assertRaisesRegex(Exception, ".*Name Relu is not found among model outputs.*"):
                convert_model(tf_net_path, output='Relu')

            tf_net_path = TestComplexParams.create_tf_model(temp_dir)

            with self.assertRaisesRegex(Exception, ".*Name Relu2 is not found among model inputs.*"):
                convert_model(tf_net_path, input='Relu2')
            with self.assertRaisesRegex(Exception, ".*Name Relu2 is not found among model outputs.*"):
                convert_model(tf_net_path, output='Relu2')