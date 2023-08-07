# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.runtime as ov
import os
import pytest
import tempfile
import unittest
from openvino.runtime import Model, Layout, PartialShape, Shape, layout_helpers, Type, Dimension

from common.mo_convert_test_class import CommonMOConvertTest
from common.tf_layer_test_class import save_to_pb


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
        {'params_test': {'output': ["Sigmoid_0", "Sigmoid_2"]},
         'params_ref': {'output': "Sigmoid_0,Sigmoid_2"}},
        {'params_test': {'output': ["Sigmoid_0"]},
         'params_ref': {'output': "Sigmoid_0"}},
        {'params_test': {'input': [PartialShape([2, 3, 4]), [2, 3, 4], [Dimension(2), Dimension(3), Dimension(4)]]},
         'params_ref': {'input_shape': "[2,3,4],[2,3,4],[2,3,4]", 'input': 'Input1,Input2,Input3'}},
        {'params_test': {'input': [PartialShape([1, 3, -1, -1]), [1, 3, -1, -1]]},
         'params_ref': {'input_shape': "[1,3,?,?],[1,3,?,?]", 'input': 'Input1,Input2'}},
        {'params_test': {'input': [(2, 3, 4), [2, 3, 4], (Dimension(2), Dimension(3), Dimension(4))]},
         'params_ref': {'input_shape': "[2,3,4],[2,3,4],[2,3,4]", 'input': 'Input1,Input2,Input3'}},
        {'params_test': {'input': {"Input1": PartialShape([2, 3, 4]), "Input2": [2, 3, 4],
                                   "Input3": [Dimension(2), Dimension(3), Dimension(4)]}},
         'params_ref': {'input_shape': "[2,3,4],[2,3,4],[2,3,4]", 'input': 'Input1,Input2,Input3'}},
        {'params_test': {'input': {"Input2": [1, -1, -1, -1],
                                   "Input3": [Dimension(1), Dimension(-1), Dimension(-1), Dimension(-1)]}},
         'params_ref': {'input_shape': "[1,?,?,?],[1,?,?,?]", 'input': 'Input2,Input3'}},
        {'params_test': {'input': [np.int32, Type(np.int32), np.int32]},
         'params_ref': {'input': 'Input1{i32},Input2{i32},Input3{i32}'}},
        {'params_test': {'input': [ov.Type.f32, ov.Type.f32]},
         'params_ref': {'input': 'Input1{f32},Input2{f32}'}},
        {'params_test': {'input': [([1, 3, -1, -1], ov.Type.i32), ov.Type.i32, ov.Type.i32]},
         'params_ref': {'input': 'Input1[1,3,?,?]{i32},Input2{i32},Input3{i32}'}}
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_mo_convert_tf_model(self, params, ie_device, precision, ir_version,
                                 temp_dir, use_new_frontend, use_old_api):
        tf_net_path = self.create_tf_model(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

    test_data = [
        {'params_test': {'input': {"Input": ([3, 2], ov.Type.i32)}},
         'params_ref': {'input': "Input[3,2]{i32}"}},
        {'params_test': {'input': {"Input": ov.Type.i32}},
         'params_ref': {'input': "Input{i32}"}},
        {'params_test': {'input': {"Input": [3, 2]}},
         'params_ref': {'input': "Input[3,2]"}},
        {'params_test': {'input': (3, 2)},
         'params_ref': {'input': "Input[3,2]"}},
        {'params_test': {'input': (3, Dimension(2))},
         'params_ref': {'input': "Input[3,2]"}},
        {'params_test': {'input': [3, 2]},
         'params_ref': {'input': "Input[3 2]"}},
        {'params_test': {'input': [Dimension(3, 10), 2]},
         'params_ref': {'input': "Input[3..10 2]"}},
        {'params_test': {'input': (-1, 10)},
         'params_ref': {'input': "Input[?,10]"}},
        {'params_test': {'input': PartialShape([-1, 10])},
         'params_ref': {'input': "Input[?,10]"}},
        {'params_test': {'input': np.int32},
         'params_ref': {'input': "Input{i32}"}},
        {'params_test': {'input': (np.int32, [1, 2, 3])},
         'params_ref': {'input': "Input[1,2,3]{i32}"}},
        {'params_test': {'input': [Dimension(3, 10), 10, -1]},
         'params_ref': {'input': 'Input[3..10,10,?]'}},
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mo_convert_tf_model_single_input_output(self, params, ie_device, precision, ir_version,
                                                     temp_dir, use_new_frontend, use_old_api):
        tf_net_path = self.create_tf_model_single_input_output(temp_dir)

        test_params = params['params_test']
        ref_params = params['params_ref']
        test_params.update({'input_model': tf_net_path})
        ref_params.update({'input_model': tf_net_path})
        self._test(temp_dir, test_params, ref_params)

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