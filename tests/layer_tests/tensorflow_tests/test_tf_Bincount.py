# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()

class TestBincount(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']

        inputs_data = {}
        inputs_data['x:0'] = rng.integers(0, 8, x_shape).astype(np.int32)

        if 'w:0' in inputs_info:
            w_shape = inputs_info['w:0']
            inputs_data['w:0'] = rng.uniform(-2.0, 2.0, w_shape).astype(self.weights_type)

        return inputs_data

    def create_bincount_net(self, input_shape, size, weights, weights_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(np.int32, input_shape, 'x')
            s = tf.constant(size)
            self.weights_type = weights_type
            if weights is not None:
              w = tf.compat.v1.placeholder(weights_type, input_shape, 'w')
            else:
              w = tf.constant([], dtype=weights_type)

            tf.raw_ops.Bincount(arr=x, size=s, weights=w)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data = [
        # with no weights
        dict(input_shape=[], size=1, weights=None, weights_type=np.float32),
        dict(input_shape=[2], size=2, weights=None, weights_type=np.float64),
        dict(input_shape=[1,3], size=3, weights=None, weights_type=np.int32),
        dict(input_shape=[3,1,4], size=4, weights=None, weights_type=np.int64),


        # with weights
        dict(input_shape=[], size=1, weights=True, weights_type=np.float32),
        dict(input_shape=[2], size=2, weights=True, weights_type=np.float64),
        dict(input_shape=[1,3], size=3, weights=True, weights_type=np.int32),
        dict(input_shape=[3,1,4], size=4, weights=True, weights_type=np.int64),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_bincount(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bincount_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)