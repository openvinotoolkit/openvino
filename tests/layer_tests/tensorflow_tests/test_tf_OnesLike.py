# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestOnesLike(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        x_shape = inputs_info['x:0']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['x:0'] = rng.integers(-10, 10, x_shape).astype(self.x_type)
        return inputs_data

    def create_ones_like_net(self, x_shape, x_type):
        self.x_type = x_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.dtypes.as_dtype(x_type), x_shape, 'x')
            tf.raw_ops.OnesLike(x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[], x_type=np.float32),
        dict(x_shape=[2], x_type=np.int32),
        dict(x_shape=[2, 3, 4], x_type=np.float32),
        dict(x_shape=[1, 4, 3, 1], x_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_ones_like(self, params, ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        self._test(*self.create_ones_like_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestComplexOnesLike(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'x_real:0' in inputs_info
        assert 'x_imag:0' in inputs_info
        x_real_shape = inputs_info['x_real:0']
        x_imag_shape = inputs_info['x_imag:0']
        inputs_data = {}
        inputs_data['x_real:0'] = 4 * rng.random(x_real_shape).astype(self.x_type) - 2
        inputs_data['x_imag:0'] = 4 * rng.random(x_imag_shape).astype(self.x_type) - 2
        return inputs_data

    def create_complex_ones_like_net(self, x_shape, x_type):
        self.x_type = x_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x_real = tf.compat.v1.placeholder(tf.dtypes.as_dtype(x_type), x_shape, 'x_real')
            x_imag = tf.compat.v1.placeholder(tf.dtypes.as_dtype(x_type), x_shape, 'x_imag')
            x_complex = tf.raw_ops.Complex(real=x_real, imag=x_imag)
            ones_like = tf.raw_ops.OnesLike(x=x_complex)
            real = tf.raw_ops.Real(input=ones_like)
            img = tf.raw_ops.Imag(input=ones_like)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[], x_type=np.float32),
        dict(x_shape=[2], x_type=np.float32),
        dict(x_shape=[2, 3, 4], x_type=np.float32),
        dict(x_shape=[1, 4, 3, 1], x_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_ones_like(self, params, ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        self._test(*self.create_complex_ones_like_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
