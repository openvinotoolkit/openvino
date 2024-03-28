# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
import numpy as np
from common.tf_layer_test_class import CommonTFLayerTest

class TestTFROund(CommonTFLayerTest):
    def create_tf_round_net(self, input_shape, input_type):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            round = tf.raw_ops.Round(input = input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
            dict(input_shape=[6], input_type=tf.float32),
            dict(input_shape=[2, 5, 3], input_type=tf.int32),
            dict(input_shape=[10, 5, 1, 5], input_type=tf.float32),
        ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_slice_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_round_net(**params),
                ie_device, precision, ir_version, temp_dir=temp_dir,
                use_legacy_frontend=use_legacy_frontend)
    
class TestTFComplexRound(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_real:1' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4* rng.random(param_real_shape_1).astype(np.float32)-2
        inputs_data['param_imag:0'] = 4* rng.random(param_imag_shape_1).astype(np.float32)-2
        return inputs_data
        

    def create_complex_round_net(self, input_shape, input_type, begin_value, size_value):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            real_input = tf.compat.v1.placeholder(input_type, input_shape, 'real_input')
            imag_input = tf.compat.v1.placeholder(input_type, input_shape, 'imag_input')
            complex = tf.raw_ops.Complex(real= real_input, imag = imag_input)
            round = tf.raw_ops.Round(complex)
            real = tf.raw_ops.Real(input = round)
            imag = tf.raw_ops.Imag(input = round)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net
        
    test_data_basic = [
    dict(input_shape=[6], input_type=tf.float32),
    dict(input_shape=[2, 5, 3], input_type=tf.int32),
    dict(input_shape=[10, 5, 1, 5], input_type=tf.float32),
    dict(input_shape=[3, 3], input_type=tf.float32),
    dict(input_shape=[4, 4, 4], input_type=tf.float32),
    dict(input_shape=[5, 5, 5, 5], input_type=tf.float32),
]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly

    def test_slice_complex(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        inputs_info = self._prepare_input(params)
        self._test(self.create_complex_round_net(*params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)