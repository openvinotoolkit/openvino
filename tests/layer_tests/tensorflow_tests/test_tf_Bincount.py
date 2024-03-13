# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestBincount(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']

        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(0, np.max(x_shape), x_shape)

        if 'w:0' in inputs_info:
            w_shape = inputs_info['w:0']
            inputs_data['w:0'] = np.random.rand(*w_shape).astype(self.weights_type)

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

    def gen_data(weights, weights_type):
      input_shape = tuple(np.random.randint(1, 10, size=np.random.randint(1, 5)))
      size = np.random.randint(1, np.max(input_shape))
      return dict(input_shape=input_shape, size=size, weights=weights, weights_type=weights_type)
  
    test_data = [
        # with no weights
        gen_data(None, np.float32),
        gen_data(None, np.float64),
        gen_data(None, np.int32),
        gen_data(None, np.int64),

        # with weights
        gen_data(True, np.float32),
        gen_data(True, np.float64),
        gen_data(True, np.int32),
        gen_data(True, np.int64),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_bincount(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bincount_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)