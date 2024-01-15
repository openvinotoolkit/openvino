# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestIdentity(CommonTFLayerTest):
    def create_identity_net(self, input_shape, identity_op):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            relu = tf.raw_ops.Relu(features=input)
            identity_op(input=relu, name="identity")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2], identity_op=tf.raw_ops.Identity),
        dict(input_shape=[2, 3], identity_op=tf.raw_ops.PreventGradient),
        dict(input_shape=[], identity_op=tf.raw_ops.Snapshot),
        dict(input_shape=[1, 2, 3], identity_op=tf.raw_ops.StopGradient)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_identity_basic(self, params, ie_device, precision, ir_version, temp_dir,
                            use_new_frontend):
        self._test(*self.create_identity_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
