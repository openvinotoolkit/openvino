# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestReciprocal(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(1, 30, x_shape).astype(np.float32)

        return inputs_data

    def create_reciprocal_net(self, x_shape, x_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            tf.raw_ops.Reciprocal(x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[], x_type=np.float32),
        dict(x_shape=[2, 3], x_type=np.float32),
        dict(x_shape=[4, 1, 3], x_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_reciprocal_basic(self, params, ie_device, precision, ir_version, temp_dir,
                              use_legacy_frontend):
        self._test(*self.create_reciprocal_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

class TestComplexReciprocal(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real_1:0' in inputs_info
        assert 'param_imag_1:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real_1:0']
        param_imag_shape_1 = inputs_info['param_imag_1:0']
        inputs_data = {}
        inputs_data['param_real_1:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag_1:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        
        return inputs_data

    def create_complex_reciprocal_net(self, x_shape,x_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real1 = tf.compat.v1.placeholder(np.float32, x_shape, 'param_real_1')
            param_imag1 = tf.compat.v1.placeholder(np.float32, x_shape, 'param_imag_1')
            complex_x = tf.raw_ops.Complex(real=param_real1, imag=param_imag1)
            reciprocal = tf.raw_ops.Reciprocal(x=complex_x)
            real = tf.raw_ops.Real(input=reciprocal)
            img = tf.raw_ops.Imag(input=reciprocal)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None
        
    test_data_basic = [
        dict(x_shape=[], x_type=np.float32),
        dict(x_shape=[2, 3], x_type=np.float32),
        dict(x_shape=[4, 1, 3], x_type=np.float32),
    ]    

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_reciprocal(self, params, ie_device, precision, ir_version, temp_dir,
                                use_legacy_frontend):
        self._test(*self.create_complex_reciprocal_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)