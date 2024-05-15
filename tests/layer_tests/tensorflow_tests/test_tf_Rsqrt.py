# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestRsqrt(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(1, 256, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_rsqrt_net(self, shape, ir_version, use_legacy_frontend):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, shape, 'Input')

            tf.math.rsqrt(input, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_precommit = [dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_rsqrt_precommit(self, params, ie_device, precision, ir_version, temp_dir,
                             use_legacy_frontend):
        self._test(*self.create_rsqrt_net(**params, ir_version=ir_version,
                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data = [dict(shape=[1]),
                 pytest.param(dict(shape=[1, 224]), marks=pytest.mark.precommit),
                 dict(shape=[1, 3, 224]),
                 dict(shape=[1, 3, 100, 224]),
                 dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_rsqrt(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_rsqrt_net(**params, ir_version=ir_version,
                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

# Test for complex numbers
class TestComplexRsqrt(CommonTFLayerTest):
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

    def create_complex_rsqrt_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            transpose = tf.raw_ops.Rsqrt(x=complex , name='Rsqrt')
            real = tf.raw_ops.Real(input=transpose)
            img = tf.raw_ops.Imag(input=transpose)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 6]),
        dict(input_shape=[2, 4, 5]),
        dict(input_shape=[1])
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_rsqrt(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(
            *self.create_complex_rsqrt_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_legacy_frontend=use_legacy_frontend)

