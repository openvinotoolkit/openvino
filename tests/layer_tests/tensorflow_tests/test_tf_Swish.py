# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(325)


class TestSwish(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(-5.0, 5.0, x_shape).astype(self.input_type)
        return inputs_data

    def create_swish_net(self, input_type, input_shape):
        self.input_type = input_type
        self.input_shape = input_shape
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            # increment helps to avoid IdentityN fused to Parameter
            swish = tf.compat.v1.nn.swish(x + 1)
            tf.identity(swish, name='swish')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64])
    @pytest.mark.parametrize("input_shape", [[], [2], [2, 3], [2, 3, 4]])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_swish(self, input_type, input_shape,
                   ie_device, precision, ir_version, temp_dir):
        custom_eps = None
        if input_type == np.float16:
            custom_eps = 3 * 1e-3
        self._test(*self.create_swish_net(input_type, input_shape),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   custom_eps=custom_eps)
