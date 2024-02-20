# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestLeakyRelu(CommonTFLayerTest):
    def create_leaky_relu_net(self, x_shape, alpha_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            if alpha_value is None:
                tf.raw_ops.LeakyRelu(features=x)
            else:
                tf.raw_ops.LeakyRelu(features=x, alpha=alpha_value)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        pytest.param(dict(x_shape=[], alpha_value=0.3), marks=pytest.mark.xfail(reason="98673")),
        dict(x_shape=[2, 3], alpha_value=None),
        dict(x_shape=[3, 4, 2], alpha_value=3),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_leaky_relu_basic(self, params, ie_device, precision, ir_version, temp_dir,
                              use_new_frontend):
        self._test(*self.create_leaky_relu_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
