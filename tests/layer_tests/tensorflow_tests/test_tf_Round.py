# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestRound(CommonTFLayerTest):
    def create_tf_round_net(self, input_shape, input_type):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Round(x=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_basic = [
        dict(input_shape=shape, input_type=type) 
        for shape in [[6], [2, 5, 3], [10, 5, 1, 5]]
        for type in [tf.float32, tf.int32, tf.int64, tf.float64]
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_round_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_round_net(**params),
                ie_device, precision, ir_version, temp_dir=temp_dir,
                use_legacy_frontend=use_legacy_frontend)

class TestComplexRound(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        return inputs_data

    def create_complex_round_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(tf.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(tf.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            # Round the real and imaginary parts separately
            rounded_real = tf.raw_ops.Round(x = tf.raw_ops.Real(input=complex))
            rounded_imag = tf.raw_ops.Round(x = tf.raw_ops.Imag(input=complex))

            # Combine the rounded parts into a rounded complex tensor
            rounded_complex = tf.raw_ops.Complex(real=rounded_real, imag=rounded_imag)

            # Since OV does not support complex tensors on output of a model, we need to extract real and imag parts
            real = tf.raw_ops.Real(input=rounded_complex)
            imag = tf.raw_ops.Imag(input=rounded_complex)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_complex = [
            dict(input_shape=[6]),
            dict(input_shape=[2, 5, 3]),
            dict(input_shape=[10, 5, 1, 5]),
        ]

    @pytest.mark.parametrize("params", test_data_complex)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_round(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_complex_round_net(**params),
                ie_device, precision, ir_version, temp_dir=temp_dir,
                use_legacy_frontend=use_legacy_frontend)
