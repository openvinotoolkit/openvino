# Copyright (C) 2018-2024 Intel Corporation
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
    @pytest.mark.precommit_tf_fe
    def test_reverse_v2_basic(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reverse_v2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
        
class TestReverseV2Complex(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        assert 'axis:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        axis_shape = inputs_info['axis:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        inputs_data['axis:0'] = rng.integers(low=0, high=min(param_real_shape.ndim, param_imag_shape.ndim), size=axis_shape).astype(np.int32)
        
        return inputs_data
    
    def create_reverse_v2_net_complex(self, shape, axis):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_real = tf.compat.v1.placeholder(np.float32, shape, 'RealInput')
            input_imag = tf.compat.v1.placeholder(np.float32, shape, 'ImaginaryInput')
            complex_input = tf.complex(input_real, input_imag)
            reversed = tf.raw_ops.ReverseV2(tensor=complex_input, axis=axis, name='reverse')
            reversed_real = tf.raw_ops.Real(reversed)
            reversed_imag = tf.raw_ops.Real(reversed)
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
    @pytest.mark.precommit_tf_fe
    def test_reverse_v2_basic_complex(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reverse_v2_net_complex(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
