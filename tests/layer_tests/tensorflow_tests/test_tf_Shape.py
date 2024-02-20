# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestShape(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        rng = np.random.default_rng()
        if self.input_type == str or self.input_type == np.str_:
            strings_dictionary = ['first', 'second sentence', ' sentence 3 three', '34ferf466 23435* ', '汉语句子',
                                  'Oferta polska',
                                  'предложение по-русски']
            inputs_data['input'] = rng.choice(strings_dictionary, input_shape)
        else:
            inputs_data['input'] = rng.integers(-10, 10, size=input_shape).astype(self.input_type)
        return inputs_data

    def create_shape_net(self, input_shape, input_type, out_type):
        self.input_type = input_type
        types_map = {
            np.float32: tf.float32,
            np.int32: tf.int32,
            str: tf.string,
            np.str_: tf.string
        }
        assert input_type in types_map, "Incorrect test case"
        tf_type = types_map[input_type]
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf_type, input_shape, 'input')
            if out_type is not None:
                tf.raw_ops.Shape(input=input, out_type=out_type)
            else:
                tf.raw_ops.Shape(input=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("input_shape", [[], [2], [3, 4], [5, 1, 2]])
    @pytest.mark.parametrize("input_type", [np.float32, np.int32, str, np.str_])
    @pytest.mark.parametrize("out_type", [tf.int32, tf.int64])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_shape_basic(self, input_shape, input_type, out_type, ie_device, precision, ir_version, temp_dir,
                         use_new_frontend):
        if input_shape == [] and out_type == tf.int64:
            pytest.skip('129100 - Hangs or segfault')
        self._test(*self.create_shape_net(input_shape=input_shape, input_type=input_type, out_type=out_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)


class TestComplexShape(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real' in inputs_info
        assert 'param_imag' in inputs_info
        param_real_shape_1 = inputs_info['param_real']
        param_imag_shape_1 = inputs_info['param_imag']
        inputs_data = {}
        inputs_data['param_real'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_shape_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            out = tf.raw_ops.Shape(input=complex, name="Shape")
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[]),
        dict(input_shape=[2, 3]),
        dict(input_shape=[2, 4, 3]),
        dict(input_shape=[2, 5, 3, 6, 8]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_shape(self, params, ie_device, precision, ir_version, temp_dir,
                           use_new_frontend):
        self._test(
            *self.create_complex_shape_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend)
