# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestReverse(CommonTFLayerTest):
    def create_reverse_net(self, shape, dims):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            tf.raw_ops.Reverse(tensor=x, dims=dims, name='reverse')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(shape=[4], dims=[True]),
        dict(shape=[3, 2], dims=[False, True]),
        dict(shape=[4, 2, 3], dims=[False, True, False]),
        dict(shape=[1, 2, 4, 3], dims=[True, False, False, False]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_reverse_basic(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reverse_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

class TestReverseComplex(CommonTFLayerTest):
    def _prepare_input_reverse(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        assert 'dims:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        dims_shape = inputs_info['dims:0']
    
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        inputs_data['dims:0'] = rng.choice([True, False], size=dims_shape).astype(np.bool_)
    
        return inputs_data
    
    def create_reverse_net_complex(self, shape, dims):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_real = tf.compat.v1.placeholder(np.float32, shape, 'RealInput')
            input_imag = tf.compat.v1.placeholder(np.float32, shape, 'ImaginaryInput')
            complex_input = tf.complex(input_real, input_imag)
            reversed = tf.raw_ops.Reverse(tensor=complex_input, dims=dims, name='reverse')
            reversed_real = tf.raw_ops.Real(reversed)
            reversed_imag = tf.raw_ops.Real(reversed)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic_complex = [
        dict(shape=[4], dims=[True]),
        dict(shape=[3, 2], dims=[False, True]),
        dict(shape=[4, 2, 3], dims=[False, True, False]),
        dict(shape=[1, 2, 4, 3], dims=[True, False, False, False]),
    ]

    @pytest.mark.parametrize("params", test_data_basic_complex)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_reverse_basic_complex(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reverse_net_complex(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)