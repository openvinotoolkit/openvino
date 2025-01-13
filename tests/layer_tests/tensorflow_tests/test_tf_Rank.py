# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestRank(CommonTFLayerTest):
    def create_rank_net(self, input_shape, input_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Rank(input=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[], input_type=tf.float32),
        dict(input_shape=[1], input_type=tf.float32),
        dict(input_shape=[2, 6], input_type=tf.int32),
        dict(input_shape=[3, 4, 5, 6], input_type=tf.float32)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_rank_basic(self, params, ie_device, precision, ir_version, temp_dir,
                        use_legacy_frontend):
        self._test(*self.create_rank_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

class TestComplexRank(CommonTFLayerTest):
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
    
    def create_rank_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input_real = tf.compat.v1.placeholder(tf.float32, input_shape, 'param_real')
            input_imag = tf.compat.v1.placeholder(tf.float32, input_shape, 'param_imag')
            input = tf.raw_ops.Complex(real=input_real, imag=input_imag)
            tf.raw_ops.Rank(input=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[]),
        dict(input_shape=[1]),
        dict(input_shape=[2, 6]),
        dict(input_shape=[3, 4, 5, 6])
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_rank(self, params, ie_device, precision, ir_version, temp_dir,
                        use_legacy_frontend):
        self._test(*self.create_rank_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
