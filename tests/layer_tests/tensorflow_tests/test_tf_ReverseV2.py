# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestReverseV2(CommonTFLayerTest):
    def create_reverse_v2_net(self, shape, axis):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            tf.raw_ops.ReverseV2(tensor=x, axis=axis, name='reverse')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(shape=[5], axis=[0]),
        dict(shape=[3], axis=[-1]),
        dict(shape=[2, 3], axis=[1]),
        dict(shape=[2, 3, 5], axis=[-2]),
        dict(shape=[2, 3, 5, 7], axis=[3]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_v2_basic(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reverse_v2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)


class TestComplexReverseV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_reverse_v2_net(self, shape, axis):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            transpose = tf.raw_ops.ReverseV2(tensor=complex, axis=axis, name='reverse')
            tf.raw_ops.Real(input=transpose)
            tf.raw_ops.Imag(input=transpose)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(shape=[5], axis=[0]),
        dict(shape=[3], axis=[-1]),
        dict(shape=[2, 3], axis=[1]),
        dict(shape=[2, 3, 5], axis=[-2]),
        dict(shape=[2, 3, 5, 7], axis=[3]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_reverse_v2_(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_complex_reverse_v2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
